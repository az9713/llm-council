"""Iterative Refinement Loop for LLM Council.

This module implements an iterative critique-and-revise cycle where:
1. Council models critique the chairman's synthesis
2. Chairman revises based on valid critiques
3. Process repeats until quality converges or max iterations reached

The refinement loop helps catch errors, fill gaps, and improve clarity
through structured feedback from the council.
"""

from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional
from .openrouter import query_models_parallel, query_model, query_model_streaming
from .config_api import get_council_models, get_chairman_model
from .pricing import calculate_cost


# Default configuration for refinement
DEFAULT_MAX_ITERATIONS = 2
DEFAULT_MIN_CRITIQUES_FOR_REVISION = 2  # Minimum number of substantive critiques to trigger revision

# Phrases indicating non-substantive (positive) feedback
NON_SUBSTANTIVE_PHRASES = [
    "no issues",
    "looks good",
    "well done",
    "comprehensive",
    "nothing to add",
    "excellent",
    "no changes needed",
    "no major issues",
    "well structured",
    "accurate",
    "complete",
    "thorough",
    "no significant",
    "no notable",
    "satisfactory",
    "adequate",
    "sufficient",
]


def is_substantive_critique(critique_text: str) -> bool:
    """
    Determine if a critique contains substantive feedback.

    Non-substantive critiques are those that only praise the response
    without identifying specific improvements.

    Args:
        critique_text: The critique text from a model

    Returns:
        True if the critique contains substantive (actionable) feedback
    """
    if not critique_text:
        return False

    text_lower = critique_text.lower()

    # Check if it's just praise without substance
    for phrase in NON_SUBSTANTIVE_PHRASES:
        if phrase in text_lower:
            # Found a non-substantive phrase, but check if there's also critique
            # Look for indicators of actual issues being raised
            issue_indicators = [
                "however",
                "but",
                "could",
                "should",
                "missing",
                "unclear",
                "incorrect",
                "error",
                "improve",
                "add",
                "expand",
                "clarify",
                "consider",
                "issue",
                "problem",
                "concern",
            ]
            has_issues = any(ind in text_lower for ind in issue_indicators)
            if not has_issues:
                return False

    # Check minimum length - very short responses are likely non-substantive
    if len(critique_text.strip()) < 50:
        return False

    return True


def count_substantive_critiques(critiques: List[Dict[str, Any]]) -> int:
    """
    Count how many critiques contain substantive feedback.

    Args:
        critiques: List of critique dicts with 'model' and 'critique' keys

    Returns:
        Number of substantive critiques
    """
    count = 0
    for c in critiques:
        if is_substantive_critique(c.get("critique", "")):
            count += 1
    return count


def create_critique_prompt(question: str, current_draft: str) -> List[Dict[str, str]]:
    """
    Create the prompt for council models to critique a draft.

    Args:
        question: The original user question
        current_draft: The current synthesis draft

    Returns:
        Message list for the critique query
    """
    return [
        {
            "role": "system",
            "content": """You are a critical reviewer helping improve answer quality.
Your job is to identify specific issues and suggest improvements.
Be constructive but thorough - point out real problems, not nitpicks.
If the answer is genuinely good, say so briefly."""
        },
        {
            "role": "user",
            "content": f"""Original question: {question}

Current answer draft:
{current_draft}

Please review this answer and provide constructive criticism:

1. **Accuracy**: Are there any factual errors or inaccuracies?
2. **Completeness**: Is anything important missing or incomplete?
3. **Clarity**: Are there parts that could be explained more clearly?
4. **Helpfulness**: What would make this answer more useful?

Be specific and actionable in your feedback. If the answer is already good, say "No major issues" and explain briefly why it's satisfactory."""
        }
    ]


def create_revision_prompt(question: str, current_draft: str, critiques: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Create the prompt for chairman to revise based on critiques.

    Args:
        question: The original user question
        current_draft: The current synthesis draft
        critiques: List of critique dicts with 'model' and 'critique' keys

    Returns:
        Message list for the revision query
    """
    # Format critiques into readable text
    feedback_parts = []
    for i, c in enumerate(critiques, 1):
        model_name = c.get("model", f"Reviewer {i}").split("/")[-1]
        critique_text = c.get("critique", "No feedback provided")
        feedback_parts.append(f"**{model_name}:**\n{critique_text}")

    feedback_text = "\n\n".join(feedback_parts)

    return [
        {
            "role": "system",
            "content": """You are improving an answer based on peer feedback.
Address valid critiques while maintaining what was already good.
Keep the same overall structure if it was effective.
Be concise - don't add unnecessary length."""
        },
        {
            "role": "user",
            "content": f"""Original question: {question}

Your previous answer:
{current_draft}

Council feedback:
{feedback_text}

Please revise your answer to address the valid critiques above.
- Keep what's already good
- Fix identified issues
- Add missing information if needed
- Improve clarity where suggested

Provide your revised answer:"""
        }
    ]


async def collect_critiques(
    question: str,
    current_draft: str,
    models: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Collect critiques from council models for a draft.

    Args:
        question: The original user question
        current_draft: The current synthesis draft
        models: Optional list of models to use (defaults to council models)

    Returns:
        Tuple of (critiques list, costs dict)
    """
    if models is None:
        models = get_council_models()

    prompt = create_critique_prompt(question, current_draft)

    # Query models in parallel
    results = await query_models_parallel(models, prompt)

    critiques = []
    total_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total": 0}

    for model, result in results.items():
        if result and result.get("content"):
            critique = {
                "model": model,
                "critique": result["content"],
                "is_substantive": is_substantive_critique(result["content"]),
                "usage": result.get("usage", {}),
                "cost": result.get("cost", {}),
            }
            critiques.append(critique)

            # Aggregate costs
            if result.get("cost"):
                total_cost["prompt_tokens"] += result["cost"].get("prompt_tokens", 0)
                total_cost["completion_tokens"] += result["cost"].get("completion_tokens", 0)
                total_cost["total"] += result["cost"].get("total", 0)

    return critiques, total_cost


async def generate_revision(
    question: str,
    current_draft: str,
    critiques: List[Dict[str, Any]],
    chairman_model: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a revision based on critiques.

    Args:
        question: The original user question
        current_draft: The current synthesis draft
        critiques: List of critique dicts
        chairman_model: Optional chairman model override

    Returns:
        Tuple of (revised text, cost dict)
    """
    if chairman_model is None:
        chairman_model = get_chairman_model()

    prompt = create_revision_prompt(question, current_draft, critiques)

    result = await query_model(chairman_model, prompt)

    if result and result.get("content"):
        return result["content"], result.get("cost", {})

    # Fallback to original draft if revision fails
    return current_draft, {}


async def generate_revision_streaming(
    question: str,
    current_draft: str,
    critiques: List[Dict[str, Any]],
    chairman_model: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate a revision with streaming for real-time display.

    Yields events:
    - revision_token: Token from chairman during revision
    - revision_complete: Revision finished with full text and costs
    - revision_error: Error during revision

    Args:
        question: The original user question
        current_draft: The current synthesis draft
        critiques: List of critique dicts
        chairman_model: Optional chairman model override
    """
    if chairman_model is None:
        chairman_model = get_chairman_model()

    prompt = create_revision_prompt(question, current_draft, critiques)

    full_response = ""
    usage = {}

    try:
        async for event in query_model_streaming(chairman_model, prompt):
            if event["type"] == "token":
                full_response += event["content"]
                yield {
                    "type": "revision_token",
                    "content": event["content"],
                    "model": chairman_model,
                }
            elif event["type"] == "complete":
                full_response = event.get("content", full_response)
                usage = event.get("usage", {})
                cost = event.get("cost", {})
                yield {
                    "type": "revision_complete",
                    "content": full_response,
                    "model": chairman_model,
                    "usage": usage,
                    "cost": cost,
                }
            elif event["type"] == "error":
                yield {
                    "type": "revision_error",
                    "error": event.get("error", "Unknown error"),
                    "model": chairman_model,
                }
    except Exception as e:
        yield {
            "type": "revision_error",
            "error": str(e),
            "model": chairman_model,
        }


async def run_refinement_loop(
    question: str,
    initial_draft: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    min_critiques_for_revision: int = DEFAULT_MIN_CRITIQUES_FOR_REVISION,
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full iterative refinement loop (non-streaming).

    Args:
        question: The original user question
        initial_draft: The initial chairman synthesis
        max_iterations: Maximum number of critique-revise cycles
        min_critiques_for_revision: Minimum substantive critiques to continue
        council_models: Optional council models override
        chairman_model: Optional chairman model override

    Returns:
        Dict containing:
        - iterations: List of iteration details
        - final_response: The final refined response
        - total_iterations: Number of iterations performed
        - total_cost: Aggregated costs across all iterations
        - converged: Whether refinement converged (vs hit max iterations)
    """
    iterations = []
    current_draft = initial_draft
    total_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total": 0}
    converged = False

    for i in range(max_iterations):
        # Collect critiques
        critiques, critique_cost = await collect_critiques(
            question, current_draft, council_models
        )

        # Update total cost
        total_cost["prompt_tokens"] += critique_cost["prompt_tokens"]
        total_cost["completion_tokens"] += critique_cost["completion_tokens"]
        total_cost["total"] += critique_cost["total"]

        # Count substantive critiques
        substantive_count = count_substantive_critiques(critiques)

        iteration_data = {
            "iteration": i + 1,
            "draft_before": current_draft,
            "critiques": critiques,
            "substantive_critique_count": substantive_count,
            "critique_cost": critique_cost,
        }

        # Check if we should stop (not enough substantive critiques)
        if substantive_count < min_critiques_for_revision:
            iteration_data["stopped"] = True
            iteration_data["stop_reason"] = f"Only {substantive_count} substantive critiques (need {min_critiques_for_revision})"
            iterations.append(iteration_data)
            converged = True
            break

        # Generate revision
        revised_draft, revision_cost = await generate_revision(
            question, current_draft, critiques, chairman_model
        )

        # Update total cost
        total_cost["prompt_tokens"] += revision_cost.get("prompt_tokens", 0)
        total_cost["completion_tokens"] += revision_cost.get("completion_tokens", 0)
        total_cost["total"] += revision_cost.get("total", 0)

        iteration_data["revision"] = revised_draft
        iteration_data["revision_cost"] = revision_cost
        iterations.append(iteration_data)

        current_draft = revised_draft

    return {
        "iterations": iterations,
        "final_response": current_draft,
        "total_iterations": len(iterations),
        "total_cost": total_cost,
        "converged": converged,
        "max_iterations": max_iterations,
    }


async def run_refinement_loop_streaming(
    question: str,
    initial_draft: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    min_critiques_for_revision: int = DEFAULT_MIN_CRITIQUES_FOR_REVISION,
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the iterative refinement loop with streaming for real-time display.

    Yields events:
    - refinement_start: Refinement loop beginning
    - iteration_start: New iteration beginning
    - critiques_start: Council critique collection starting
    - critique_complete: A single model's critique received
    - critiques_complete: All critiques collected for iteration
    - revision_start: Chairman revision starting
    - revision_token: Token from chairman during revision
    - revision_complete: Revision finished
    - iteration_complete: Full iteration finished
    - refinement_complete: Refinement loop finished
    - refinement_converged: Refinement stopped early (converged)

    Args:
        question: The original user question
        initial_draft: The initial chairman synthesis
        max_iterations: Maximum number of critique-revise cycles
        min_critiques_for_revision: Minimum substantive critiques to continue
        council_models: Optional council models override
        chairman_model: Optional chairman model override
    """
    if council_models is None:
        council_models = get_council_models()
    if chairman_model is None:
        chairman_model = get_chairman_model()

    yield {
        "type": "refinement_start",
        "max_iterations": max_iterations,
        "council_models": council_models,
        "chairman_model": chairman_model,
    }

    iterations = []
    current_draft = initial_draft
    total_cost = {"prompt_tokens": 0, "completion_tokens": 0, "total": 0}
    converged = False

    for i in range(max_iterations):
        yield {
            "type": "iteration_start",
            "iteration": i + 1,
            "draft": current_draft,
        }

        # Collect critiques
        yield {"type": "critiques_start", "iteration": i + 1, "model_count": len(council_models)}

        critiques, critique_cost = await collect_critiques(
            question, current_draft, council_models
        )

        # Emit individual critique completions
        for critique in critiques:
            yield {
                "type": "critique_complete",
                "iteration": i + 1,
                "model": critique["model"],
                "critique": critique["critique"],
                "is_substantive": critique["is_substantive"],
                "cost": critique.get("cost", {}),
            }

        # Update total cost
        total_cost["prompt_tokens"] += critique_cost["prompt_tokens"]
        total_cost["completion_tokens"] += critique_cost["completion_tokens"]
        total_cost["total"] += critique_cost["total"]

        substantive_count = count_substantive_critiques(critiques)

        yield {
            "type": "critiques_complete",
            "iteration": i + 1,
            "critiques": critiques,
            "substantive_count": substantive_count,
            "total_cost": critique_cost,
        }

        iteration_data = {
            "iteration": i + 1,
            "draft_before": current_draft,
            "critiques": critiques,
            "substantive_critique_count": substantive_count,
            "critique_cost": critique_cost,
        }

        # Check if we should stop
        if substantive_count < min_critiques_for_revision:
            iteration_data["stopped"] = True
            iteration_data["stop_reason"] = f"Only {substantive_count} substantive critiques (need {min_critiques_for_revision})"
            iterations.append(iteration_data)
            converged = True

            yield {
                "type": "refinement_converged",
                "iteration": i + 1,
                "reason": iteration_data["stop_reason"],
                "final_response": current_draft,
            }
            break

        # Generate revision with streaming
        yield {
            "type": "revision_start",
            "iteration": i + 1,
            "model": chairman_model,
        }

        revised_draft = ""
        revision_cost = {}

        async for event in generate_revision_streaming(
            question, current_draft, critiques, chairman_model
        ):
            if event["type"] == "revision_token":
                yield {
                    "type": "revision_token",
                    "iteration": i + 1,
                    "content": event["content"],
                    "model": event["model"],
                }
            elif event["type"] == "revision_complete":
                revised_draft = event["content"]
                revision_cost = event.get("cost", {})
                yield {
                    "type": "revision_complete",
                    "iteration": i + 1,
                    "content": revised_draft,
                    "model": event["model"],
                    "cost": revision_cost,
                }
            elif event["type"] == "revision_error":
                yield {
                    "type": "revision_error",
                    "iteration": i + 1,
                    "error": event["error"],
                    "model": event.get("model"),
                }
                # Use current draft as fallback
                revised_draft = current_draft

        # Update total cost
        total_cost["prompt_tokens"] += revision_cost.get("prompt_tokens", 0)
        total_cost["completion_tokens"] += revision_cost.get("completion_tokens", 0)
        total_cost["total"] += revision_cost.get("total", 0)

        iteration_data["revision"] = revised_draft
        iteration_data["revision_cost"] = revision_cost
        iterations.append(iteration_data)

        yield {
            "type": "iteration_complete",
            "iteration": i + 1,
            "revision": revised_draft,
            "total_cost": total_cost,
        }

        current_draft = revised_draft

    yield {
        "type": "refinement_complete",
        "iterations": iterations,
        "final_response": current_draft,
        "total_iterations": len(iterations),
        "total_cost": total_cost,
        "converged": converged,
        "max_iterations": max_iterations,
    }


def get_refinement_config() -> Dict[str, Any]:
    """
    Get current refinement configuration.

    Returns:
        Dict with default_max_iterations, min_critiques_for_revision, etc.
    """
    return {
        "default_max_iterations": DEFAULT_MAX_ITERATIONS,
        "min_critiques_for_revision": DEFAULT_MIN_CRITIQUES_FOR_REVISION,
        "non_substantive_phrases": NON_SUBSTANTIVE_PHRASES,
    }
