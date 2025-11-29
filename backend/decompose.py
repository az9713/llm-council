"""
Sub-Question Decomposition Module.

Implements map-reduce pattern for complex questions:
1. Decompose: Break complex question into simpler sub-questions
2. Map: Run each sub-question through a mini-council
3. Reduce: Merge/synthesize all sub-answers into final response

This is an alternative flow that replaces the normal 3-stage council process
when enabled and the question is detected as complex.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple

from .openrouter import query_model, query_models_parallel, query_model_streaming
from .config_api import get_council_models, get_chairman_model
from .pricing import aggregate_costs


# Configuration constants
DEFAULT_MAX_SUB_QUESTIONS = 5
DEFAULT_MIN_SUB_QUESTIONS = 2
COMPLEXITY_THRESHOLD = 0.6  # Confidence threshold for decomposition

# Model used for decomposition analysis (fast, good at structured output)
DECOMPOSER_MODEL = "google/gemini-2.0-flash-001"

# Complexity indicators that suggest decomposition would help
COMPLEXITY_INDICATORS = [
    "compare", "contrast", "analyze", "evaluate", "explain how",
    "what are the differences", "pros and cons", "advantages and disadvantages",
    "step by step", "multiple", "various", "several", "different aspects",
    "comprehensive", "in-depth", "detailed analysis", "break down",
    "on one hand", "on the other hand", "firstly", "secondly",
    "multi-part", "multi-faceted", "complex", "complicated"
]

# Prompt for detecting if a question needs decomposition
COMPLEXITY_DETECTION_PROMPT = """Analyze this question and determine if it would benefit from being broken down into sub-questions.

A question should be decomposed if it:
1. Asks about multiple distinct topics or aspects
2. Requires comparing/contrasting different things
3. Has multiple parts that could be answered independently
4. Would benefit from a structured, step-by-step analysis
5. Is complex enough that separate focused answers would be better than one broad answer

Question: {question}

Respond in this exact format:
SHOULD_DECOMPOSE: [YES or NO]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation of why or why not]
"""

# Prompt for decomposing a question into sub-questions
DECOMPOSITION_PROMPT = """Break down this complex question into {min_count} to {max_count} focused sub-questions.

Each sub-question should:
1. Be self-contained and answerable independently
2. Cover a distinct aspect of the original question
3. Be clear and specific
4. Together, the sub-questions should fully address the original question

Original Question: {question}

Respond with numbered sub-questions only, one per line:
1. [First sub-question]
2. [Second sub-question]
...
"""

# Prompt for council models answering a sub-question
SUB_QUESTION_PROMPT = """You are answering a focused sub-question as part of a larger analysis.

Original Question: {original_question}

Your Sub-Question: {sub_question}

Provide a focused, thorough answer to this specific sub-question. Your answer will be combined with answers to other sub-questions to form a complete response.

Be concise but comprehensive for this specific aspect.
"""

# Prompt for merging sub-answers into final response
MERGE_PROMPT = """You are synthesizing multiple focused answers into one comprehensive response.

Original Question: {question}

The question was broken into sub-questions, each answered by a council of AI models. Here are the best answers for each sub-question:

{sub_answers}

Your task:
1. Synthesize these focused answers into one coherent, comprehensive response
2. Maintain the key insights from each sub-answer
3. Ensure smooth transitions between topics
4. Remove any redundancy while keeping all important information
5. Present the final answer in a clear, well-organized format

Provide the complete, unified answer:
"""


def has_complexity_keywords(question: str) -> bool:
    """Check if question contains keywords suggesting complexity."""
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in COMPLEXITY_INDICATORS)


async def detect_complexity(question: str) -> Tuple[bool, float, str]:
    """
    Determine if a question needs decomposition.

    Args:
        question: The user's question

    Returns:
        Tuple of (should_decompose, confidence, reasoning)
    """
    # Quick heuristic check first
    has_keywords = has_complexity_keywords(question)

    # Use LLM for more nuanced detection
    messages = [
        {"role": "user", "content": COMPLEXITY_DETECTION_PROMPT.format(question=question)}
    ]

    try:
        result = await query_model(DECOMPOSER_MODEL, messages, timeout=30.0)
        if result and result.get('content'):
            content = result['content']

            # Parse the response
            should_decompose = "SHOULD_DECOMPOSE: YES" in content.upper()

            # Extract confidence
            confidence = 0.5
            if "CONFIDENCE:" in content.upper():
                try:
                    conf_line = [l for l in content.split('\n') if 'CONFIDENCE:' in l.upper()][0]
                    conf_str = conf_line.split(':')[1].strip()
                    confidence = float(conf_str.replace(',', '.'))
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5

            # Extract reasoning
            reasoning = ""
            if "REASONING:" in content.upper():
                try:
                    reasoning_idx = content.upper().find("REASONING:")
                    reasoning = content[reasoning_idx + 10:].strip()
                except (ValueError, IndexError):
                    reasoning = "Analysis complete."

            # Combine with keyword heuristic
            if has_keywords and confidence < 0.5:
                confidence = min(confidence + 0.2, 0.7)

            return (should_decompose and confidence >= COMPLEXITY_THRESHOLD, confidence, reasoning)

    except Exception as e:
        print(f"Error in complexity detection: {e}")

    # Fallback to keyword heuristic
    return (has_keywords, 0.4 if has_keywords else 0.2, "Based on keyword analysis.")


async def decompose_question(
    question: str,
    min_count: int = DEFAULT_MIN_SUB_QUESTIONS,
    max_count: int = DEFAULT_MAX_SUB_QUESTIONS
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Break a complex question into sub-questions.

    Args:
        question: The complex question to decompose
        min_count: Minimum number of sub-questions
        max_count: Maximum number of sub-questions

    Returns:
        Tuple of (list of sub-questions, cost data)
    """
    messages = [
        {"role": "user", "content": DECOMPOSITION_PROMPT.format(
            question=question,
            min_count=min_count,
            max_count=max_count
        )}
    ]

    result = await query_model(DECOMPOSER_MODEL, messages, timeout=60.0)

    if not result or not result.get('content'):
        # Fallback: return original question as single sub-question
        return [question], {"total_cost": 0}

    content = result['content']

    # Parse numbered sub-questions
    sub_questions = []
    for line in content.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove numbering prefix (e.g., "1.", "1)", "1:")
            for sep in ['.', ')', ':', '-']:
                if sep in line[:3]:
                    line = line[line.index(sep) + 1:].strip()
                    break
            if line:
                sub_questions.append(line)

    # Ensure we have valid sub-questions
    if len(sub_questions) < min_count:
        # If decomposition failed, return original question
        return [question], result.get('cost', {"total_cost": 0})

    # Limit to max count
    sub_questions = sub_questions[:max_count]

    return sub_questions, result.get('cost', {"total_cost": 0})


async def run_sub_council(
    original_question: str,
    sub_question: str,
    sub_index: int,
    models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run a mini-council for a single sub-question.

    Args:
        original_question: The original complex question
        sub_question: The specific sub-question to answer
        sub_index: Index of this sub-question (0-based)
        models: Council models to use (defaults to configured models)

    Returns:
        Dict with responses, best_response, rankings, cost
    """
    if models is None:
        models = get_council_models()

    # Build the prompt for this sub-question
    prompt = SUB_QUESTION_PROMPT.format(
        original_question=original_question,
        sub_question=sub_question
    )

    messages = [{"role": "user", "content": prompt}]

    # Query all models in parallel
    responses = await query_models_parallel(models, messages)

    # Collect valid responses
    valid_responses = []
    total_cost = {"input_cost": 0, "output_cost": 0, "total_cost": 0}

    for model, response in responses.items():
        if response and response.get('content'):
            valid_responses.append({
                "model": model,
                "response": response['content'],
                "cost": response.get('cost', {})
            })
            # Aggregate costs
            if response.get('cost'):
                total_cost['input_cost'] += response['cost'].get('input_cost', 0)
                total_cost['output_cost'] += response['cost'].get('output_cost', 0)
                total_cost['total_cost'] += response['cost'].get('total_cost', 0)

    # For simplicity, select the longest/most comprehensive response as "best"
    # In a full implementation, we could run Stage 2 ranking for each sub-question
    best_response = None
    if valid_responses:
        # Use the response with most content as a simple heuristic
        best_response = max(valid_responses, key=lambda r: len(r['response']))

    return {
        "sub_question": sub_question,
        "sub_index": sub_index,
        "responses": valid_responses,
        "best_response": best_response,
        "model_count": len(valid_responses),
        "cost": total_cost
    }


async def merge_sub_answers(
    question: str,
    sub_results: List[Dict[str, Any]],
    chairman_model: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Merge sub-answers into a final comprehensive response.

    Args:
        question: The original question
        sub_results: List of sub-council results
        chairman_model: Model to use for merging (defaults to configured chairman)

    Yields:
        Events: merge_token, merge_complete
    """
    if chairman_model is None:
        chairman_model = get_chairman_model()

    # Format sub-answers for the merge prompt
    sub_answers_text = ""
    for i, result in enumerate(sub_results, 1):
        sub_q = result.get('sub_question', f'Sub-question {i}')
        best = result.get('best_response', {})
        answer = best.get('response', 'No answer available') if best else 'No answer available'
        model = best.get('model', 'Unknown') if best else 'Unknown'

        sub_answers_text += f"\n--- Sub-Question {i}: {sub_q} ---\n"
        sub_answers_text += f"Best Answer (from {model}):\n{answer}\n"

    # Build merge prompt
    prompt = MERGE_PROMPT.format(
        question=question,
        sub_answers=sub_answers_text
    )

    messages = [{"role": "user", "content": prompt}]

    # Stream the merge response
    full_content = ""
    async for event in query_model_streaming(chairman_model, messages):
        if event['type'] == 'token':
            full_content += event['content']
            yield {
                "type": "merge_token",
                "content": event['content'],
                "model": chairman_model
            }
        elif event['type'] == 'complete':
            yield {
                "type": "merge_complete",
                "content": full_content,
                "model": chairman_model,
                "cost": event.get('cost', {}),
                "usage": event.get('usage', {})
            }
        elif event['type'] == 'error':
            yield {
                "type": "merge_error",
                "error": event.get('error', 'Unknown error'),
                "model": chairman_model
            }


async def run_decomposition_streaming(
    question: str,
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the full decomposition flow with streaming.

    This is the main orchestration function for sub-question decomposition.

    Args:
        question: The user's question
        council_models: Models to use for sub-councils
        chairman_model: Model to use for merging
        max_sub_questions: Maximum sub-questions to generate

    Yields:
        Events for the decomposition process:
        - decomposition_start: Beginning decomposition analysis
        - complexity_analyzed: Complexity analysis complete
        - decomposition_skip: Question not complex enough
        - sub_questions_generated: Sub-questions created
        - sub_council_start: Starting work on a sub-question
        - sub_council_response: A model responded to sub-question
        - sub_council_complete: Sub-question fully answered
        - all_sub_councils_complete: All sub-questions answered
        - merge_start: Beginning merge/synthesis
        - merge_token: Token from merger
        - merge_complete: Merge finished
        - decomposition_complete: Full process complete
    """
    if council_models is None:
        council_models = get_council_models()
    if chairman_model is None:
        chairman_model = get_chairman_model()

    total_cost = {"input_cost": 0, "output_cost": 0, "total_cost": 0}

    # Start decomposition process
    yield {
        "type": "decomposition_start",
        "question": question,
        "council_models": council_models,
        "chairman_model": chairman_model,
        "max_sub_questions": max_sub_questions
    }

    # Analyze complexity
    should_decompose, confidence, reasoning = await detect_complexity(question)

    yield {
        "type": "complexity_analyzed",
        "should_decompose": should_decompose,
        "confidence": confidence,
        "reasoning": reasoning
    }

    if not should_decompose:
        yield {
            "type": "decomposition_skip",
            "reason": reasoning,
            "confidence": confidence
        }
        return

    # Decompose the question
    sub_questions, decompose_cost = await decompose_question(
        question,
        min_count=DEFAULT_MIN_SUB_QUESTIONS,
        max_count=max_sub_questions
    )

    if decompose_cost.get('total_cost'):
        total_cost['total_cost'] += decompose_cost.get('total_cost', 0)

    yield {
        "type": "sub_questions_generated",
        "sub_questions": sub_questions,
        "count": len(sub_questions),
        "cost": decompose_cost
    }

    # Run sub-councils for each sub-question
    sub_results = []

    for i, sub_q in enumerate(sub_questions):
        yield {
            "type": "sub_council_start",
            "sub_index": i,
            "sub_question": sub_q,
            "total_sub_questions": len(sub_questions)
        }

        # Run the sub-council
        result = await run_sub_council(
            original_question=question,
            sub_question=sub_q,
            sub_index=i,
            models=council_models
        )

        # Aggregate cost
        if result.get('cost'):
            total_cost['input_cost'] += result['cost'].get('input_cost', 0)
            total_cost['output_cost'] += result['cost'].get('output_cost', 0)
            total_cost['total_cost'] += result['cost'].get('total_cost', 0)

        # Emit response events for each model
        for response in result.get('responses', []):
            yield {
                "type": "sub_council_response",
                "sub_index": i,
                "sub_question": sub_q,
                "model": response['model'],
                "response": response['response'],
                "cost": response.get('cost', {})
            }

        yield {
            "type": "sub_council_complete",
            "sub_index": i,
            "sub_question": sub_q,
            "best_response": result.get('best_response'),
            "model_count": result.get('model_count', 0),
            "cost": result.get('cost', {})
        }

        sub_results.append(result)

    yield {
        "type": "all_sub_councils_complete",
        "sub_results_count": len(sub_results),
        "total_responses": sum(r.get('model_count', 0) for r in sub_results)
    }

    # Merge sub-answers into final response
    yield {
        "type": "merge_start",
        "model": chairman_model,
        "sub_questions_count": len(sub_questions)
    }

    final_response = ""
    merge_cost = {}

    async for event in merge_sub_answers(question, sub_results, chairman_model):
        if event['type'] == 'merge_token':
            final_response += event['content']
            yield event
        elif event['type'] == 'merge_complete':
            merge_cost = event.get('cost', {})
            total_cost['input_cost'] += merge_cost.get('input_cost', 0)
            total_cost['output_cost'] += merge_cost.get('output_cost', 0)
            total_cost['total_cost'] += merge_cost.get('total_cost', 0)
            yield event
        elif event['type'] == 'merge_error':
            yield event
            return

    # Final completion event
    yield {
        "type": "decomposition_complete",
        "sub_questions": sub_questions,
        "sub_results": [
            {
                "sub_question": r['sub_question'],
                "best_answer": r['best_response']['response'] if r.get('best_response') else None,
                "best_model": r['best_response']['model'] if r.get('best_response') else None,
                "model_count": r.get('model_count', 0)
            }
            for r in sub_results
        ],
        "final_response": final_response,
        "chairman_model": chairman_model,
        "total_cost": total_cost
    }


def get_decomposition_config() -> Dict[str, Any]:
    """Get decomposition configuration for API."""
    return {
        "default_max_sub_questions": DEFAULT_MAX_SUB_QUESTIONS,
        "default_min_sub_questions": DEFAULT_MIN_SUB_QUESTIONS,
        "complexity_threshold": COMPLEXITY_THRESHOLD,
        "decomposer_model": DECOMPOSER_MODEL,
        "complexity_indicators": COMPLEXITY_INDICATORS[:10],  # First 10 as examples
        "description": "Breaks complex questions into sub-questions, answers each separately, then merges"
    }
