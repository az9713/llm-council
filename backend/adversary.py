"""
Adversarial Validation Module.

Implements a "devil's advocate" review where an adversary model attempts to find
flaws, errors, or weaknesses in the chairman's synthesis. If genuine issues are
found, the chairman revises the response.

This provides an additional quality assurance layer by stress-testing the synthesis.
"""

import asyncio
import re
from typing import Optional, Tuple, Dict, List, Any, AsyncGenerator

from .config_api import get_chairman_model
from .openrouter import query_model, query_model_streaming
from .pricing import aggregate_costs


# Default adversary model (fast, capable of critical analysis)
DEFAULT_ADVERSARY_MODEL = "google/gemini-2.0-flash-001"

# Phrases indicating no genuine issues found
NO_ISSUES_PHRASES = [
    "no issues",
    "no problems",
    "no errors",
    "no flaws",
    "no mistakes",
    "looks good",
    "looks correct",
    "appears correct",
    "appears accurate",
    "well-reasoned",
    "well-written",
    "comprehensive and accurate",
    "nothing wrong",
    "no concerns",
    "no objections",
    "no criticisms",
    "unable to find",
    "cannot find any",
    "can't find any",
    "could not find",
    "couldn't find",
    "no significant issues",
    "no major issues",
    "no obvious errors",
    "approve this response",
    "endorsed",
    "satisfactory",
]

# Issue severity levels
SEVERITY_CRITICAL = "critical"
SEVERITY_MAJOR = "major"
SEVERITY_MINOR = "minor"
SEVERITY_NONE = "none"


def get_adversary_model() -> str:
    """Get the adversary model to use for validation."""
    # Could be made configurable via config_api in the future
    return DEFAULT_ADVERSARY_MODEL


def has_genuine_issues(critique: str) -> Tuple[bool, str]:
    """
    Determine if the adversary critique contains genuine issues.

    Returns:
        Tuple of (has_issues: bool, severity: str)
    """
    if not critique:
        return False, SEVERITY_NONE

    critique_lower = critique.lower()

    # Check for explicit "no issues" phrases
    for phrase in NO_ISSUES_PHRASES:
        if phrase in critique_lower:
            return False, SEVERITY_NONE

    # Check for severity indicators
    if any(word in critique_lower for word in ["critical", "severe", "fatal", "incorrect", "wrong", "false", "error"]):
        return True, SEVERITY_CRITICAL

    if any(word in critique_lower for word in ["major", "significant", "important", "missing", "incomplete", "inaccurate"]):
        return True, SEVERITY_MAJOR

    if any(word in critique_lower for word in ["minor", "small", "slight", "could be improved", "consider", "suggestion"]):
        return True, SEVERITY_MINOR

    # Default: if none of the "no issues" phrases found, assume there might be issues
    # But check if the critique actually identifies something
    issue_indicators = ["issue", "problem", "flaw", "weakness", "error", "mistake", "concern", "incorrect", "missing"]
    if any(word in critique_lower for word in issue_indicators):
        return True, SEVERITY_MAJOR

    return False, SEVERITY_NONE


def create_adversary_prompt(question: str, synthesis: str) -> str:
    """
    Create the prompt for the adversary to critically review the synthesis.

    The adversary acts as a "devil's advocate" trying to find flaws.
    """
    return f"""You are a critical reviewer acting as a "devil's advocate". Your job is to carefully examine this response and try to find any flaws, errors, or weaknesses.

ORIGINAL QUESTION:
{question}

RESPONSE TO REVIEW:
{synthesis}

YOUR TASK:
1. Carefully analyze the response for factual errors, logical flaws, missing information, or misleading statements
2. Check if the response actually answers the question completely
3. Look for any oversimplifications, biases, or unsupported claims
4. Identify any security concerns or potentially harmful advice (if applicable)

If you find genuine issues, list them clearly with:
- The specific issue found
- Why it's problematic
- The severity (CRITICAL, MAJOR, or MINOR)

If the response is accurate and complete with no significant issues, simply state "No significant issues found" and briefly explain why the response is adequate.

Be rigorous but fair. Only flag genuine issues, not stylistic preferences.

YOUR CRITICAL REVIEW:"""


def create_revision_prompt(question: str, original_synthesis: str, adversary_critique: str) -> str:
    """
    Create the prompt for the chairman to revise based on adversary feedback.
    """
    return f"""You previously synthesized a response, but a critical review has identified some issues. Please revise your response to address these concerns.

ORIGINAL QUESTION:
{question}

YOUR PREVIOUS RESPONSE:
{original_synthesis}

CRITICAL REVIEW:
{adversary_critique}

YOUR TASK:
1. Carefully consider each issue raised in the critical review
2. Revise your response to address legitimate concerns
3. If any critique points are invalid, you may keep your original text for those parts
4. Maintain the overall quality and completeness of your response

Provide only your revised response, without any preamble or explanation of changes.

REVISED RESPONSE:"""


async def adversarial_review(
    question: str,
    synthesis: str,
    adversary_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Have an adversary model critically review the synthesis.

    Args:
        question: The original user question
        synthesis: The chairman's synthesis to review
        adversary_model: Optional model to use (defaults to configured adversary model)

    Returns:
        Dict with: model, critique, has_issues, severity, usage, cost
    """
    model = adversary_model or get_adversary_model()
    prompt = create_adversary_prompt(question, synthesis)

    result = await query_model(model, prompt)

    if not result:
        return {
            "model": model,
            "critique": None,
            "has_issues": False,
            "severity": SEVERITY_NONE,
            "error": "Failed to get adversary review",
            "usage": None,
            "cost": None,
        }

    critique = result.get("content", "")
    has_issues, severity = has_genuine_issues(critique)

    return {
        "model": model,
        "critique": critique,
        "has_issues": has_issues,
        "severity": severity,
        "usage": result.get("usage"),
        "cost": result.get("cost"),
    }


async def adversarial_review_streaming(
    question: str,
    synthesis: str,
    adversary_model: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream the adversary's critical review.

    Yields:
        Events: adversary_token, adversary_complete, adversary_error
    """
    model = adversary_model or get_adversary_model()
    prompt = create_adversary_prompt(question, synthesis)

    full_critique = ""
    usage = None
    cost = None

    try:
        async for event in query_model_streaming(model, prompt):
            if event["type"] == "token":
                full_critique += event["content"]
                yield {
                    "type": "adversary_token",
                    "model": model,
                    "content": event["content"],
                }
            elif event["type"] == "complete":
                usage = event.get("usage")
                cost = event.get("cost")
                full_critique = event.get("content", full_critique)
    except Exception as e:
        yield {
            "type": "adversary_error",
            "model": model,
            "error": str(e),
        }
        return

    # Analyze the complete critique
    has_issues, severity = has_genuine_issues(full_critique)

    yield {
        "type": "adversary_complete",
        "model": model,
        "critique": full_critique,
        "has_issues": has_issues,
        "severity": severity,
        "usage": usage,
        "cost": cost,
    }


async def generate_adversary_revision(
    question: str,
    original_synthesis: str,
    adversary_critique: str,
    chairman_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a revised synthesis based on adversary feedback.

    Args:
        question: The original user question
        original_synthesis: The original chairman synthesis
        adversary_critique: The adversary's critique
        chairman_model: Optional model to use (defaults to configured chairman)

    Returns:
        Dict with: model, response, usage, cost
    """
    model = chairman_model or get_chairman_model()
    prompt = create_revision_prompt(question, original_synthesis, adversary_critique)

    result = await query_model(model, prompt)

    if not result:
        return {
            "model": model,
            "response": original_synthesis,  # Fall back to original
            "revision_failed": True,
            "usage": None,
            "cost": None,
        }

    return {
        "model": model,
        "response": result.get("content", original_synthesis),
        "usage": result.get("usage"),
        "cost": result.get("cost"),
    }


async def generate_adversary_revision_streaming(
    question: str,
    original_synthesis: str,
    adversary_critique: str,
    chairman_model: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream the chairman's revision based on adversary feedback.

    Yields:
        Events: revision_token, revision_complete, revision_error
    """
    model = chairman_model or get_chairman_model()
    prompt = create_revision_prompt(question, original_synthesis, adversary_critique)

    full_response = ""
    usage = None
    cost = None

    try:
        async for event in query_model_streaming(model, prompt):
            if event["type"] == "token":
                full_response += event["content"]
                yield {
                    "type": "adversary_revision_token",
                    "model": model,
                    "content": event["content"],
                }
            elif event["type"] == "complete":
                usage = event.get("usage")
                cost = event.get("cost")
                full_response = event.get("content", full_response)
    except Exception as e:
        yield {
            "type": "adversary_revision_error",
            "model": model,
            "error": str(e),
        }
        return

    yield {
        "type": "adversary_revision_complete",
        "model": model,
        "response": full_response,
        "usage": usage,
        "cost": cost,
    }


async def run_adversarial_validation(
    question: str,
    synthesis: str,
    adversary_model: Optional[str] = None,
    chairman_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full adversarial validation flow (non-streaming).

    1. Adversary reviews synthesis
    2. If issues found, chairman revises
    3. Return final result

    Returns:
        Dict with: adversary_review, revision (if applicable), final_response,
                   issues_found, severity, total_cost
    """
    # Step 1: Adversary review
    review = await adversarial_review(question, synthesis, adversary_model)

    result = {
        "adversary_review": review,
        "issues_found": review["has_issues"],
        "severity": review["severity"],
        "revision": None,
        "final_response": synthesis,
        "validation_applied": True,
    }

    costs = []
    if review.get("cost"):
        costs.append(review["cost"])

    # Step 2: If issues found, have chairman revise
    if review["has_issues"] and review["severity"] in [SEVERITY_CRITICAL, SEVERITY_MAJOR]:
        revision = await generate_adversary_revision(
            question, synthesis, review["critique"], chairman_model
        )
        result["revision"] = revision
        result["final_response"] = revision.get("response", synthesis)
        if revision.get("cost"):
            costs.append(revision["cost"])

    result["total_cost"] = aggregate_costs(costs)

    return result


async def run_adversarial_validation_streaming(
    question: str,
    synthesis: str,
    adversary_model: Optional[str] = None,
    chairman_model: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the full adversarial validation flow with streaming.

    Yields:
        Events for the adversarial review and revision process
    """
    adv_model = adversary_model or get_adversary_model()
    chair_model = chairman_model or get_chairman_model()

    # Signal start
    yield {
        "type": "adversary_start",
        "adversary_model": adv_model,
    }

    # Step 1: Stream adversary review
    full_critique = ""
    review_result = None
    review_cost = None

    async for event in adversarial_review_streaming(question, synthesis, adv_model):
        yield event
        if event["type"] == "adversary_complete":
            review_result = event
            full_critique = event["critique"]
            review_cost = event.get("cost")

    if not review_result:
        yield {
            "type": "adversary_error",
            "error": "Failed to complete adversarial review",
        }
        return

    has_issues = review_result["has_issues"]
    severity = review_result["severity"]

    costs = []
    if review_cost:
        costs.append(review_cost)

    # Step 2: If issues found, stream revision
    revision_response = None
    if has_issues and severity in [SEVERITY_CRITICAL, SEVERITY_MAJOR]:
        yield {
            "type": "adversary_revision_start",
            "chairman_model": chair_model,
            "severity": severity,
        }

        async for event in generate_adversary_revision_streaming(
            question, synthesis, full_critique, chair_model
        ):
            yield event
            if event["type"] == "adversary_revision_complete":
                revision_response = event.get("response")
                if event.get("cost"):
                    costs.append(event["cost"])

    # Final result
    yield {
        "type": "adversary_validation_complete",
        "issues_found": has_issues,
        "severity": severity,
        "final_response": revision_response or synthesis,
        "revised": revision_response is not None,
        "total_cost": aggregate_costs(costs),
    }


def get_adversary_config() -> Dict[str, Any]:
    """Get the current adversarial validation configuration."""
    return {
        "adversary_model": get_adversary_model(),
        "no_issues_phrases": NO_ISSUES_PHRASES[:10],  # Sample
        "severity_levels": [SEVERITY_CRITICAL, SEVERITY_MAJOR, SEVERITY_MINOR, SEVERITY_NONE],
        "revision_threshold": [SEVERITY_CRITICAL, SEVERITY_MAJOR],
    }


# Severity info for UI display
SEVERITY_INFO = {
    SEVERITY_CRITICAL: {
        "label": "Critical",
        "description": "Serious errors that must be corrected",
        "color": "#dc2626",  # red-600
    },
    SEVERITY_MAJOR: {
        "label": "Major",
        "description": "Significant issues that should be addressed",
        "color": "#ea580c",  # orange-600
    },
    SEVERITY_MINOR: {
        "label": "Minor",
        "description": "Small improvements possible",
        "color": "#ca8a04",  # yellow-600
    },
    SEVERITY_NONE: {
        "label": "None",
        "description": "No issues found",
        "color": "#16a34a",  # green-600
    },
}
