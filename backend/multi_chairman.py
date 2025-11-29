"""Multi-Chairman Synthesis for LLM Council.

This module implements ensemble synthesis where multiple chairman models
independently synthesize final responses, then a supreme chairman selects
the best synthesis.

The multi-chairman approach provides:
- Diverse synthesis perspectives from different models
- Quality assurance through supreme chairman evaluation
- Reduced risk of single model failures or biases
"""

from typing import List, Dict, Any, AsyncGenerator
from .openrouter import query_models_parallel, query_model, query_model_streaming
from .config_api import get_chairman_model, get_multi_chairman_models
from .pricing import aggregate_costs


def get_supreme_chairman_model() -> str:
    """
    Get the supreme chairman model that selects the best synthesis.

    Uses the configured chairman model as the supreme chairman.

    Returns:
        Model identifier for the supreme chairman
    """
    return get_chairman_model()


async def stage3_multi_chairman_synthesis(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Stage 3A: Multiple chairmen synthesize in parallel.

    Each chairman model independently creates a synthesis based on
    the council responses and peer rankings.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        List of dicts with 'model', 'response', 'usage', and 'cost' for each chairman
    """
    # Build comprehensive context for chairmen
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are a Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query all chairman models in parallel
    chairman_models = get_multi_chairman_models()
    responses = await query_models_parallel(chairman_models, messages)

    # Format results
    syntheses = []
    for model, response in responses.items():
        if response is not None:
            syntheses.append({
                "model": model,
                "response": response.get('content', ''),
                "usage": response.get('usage', {}),
                "cost": response.get('cost', {}),
            })
        else:
            syntheses.append({
                "model": model,
                "response": "Error: Unable to generate synthesis.",
                "usage": {},
                "cost": {},
            })

    return syntheses


async def stage3_supreme_chairman_selection(
    user_query: str,
    syntheses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3B: Supreme chairman selects the best synthesis.

    The supreme chairman evaluates all syntheses and selects the best one,
    providing reasoning for the selection.

    Args:
        user_query: The original user query
        syntheses: List of synthesis results from multi-chairman step

    Returns:
        Dict with 'model', 'response', 'selected_synthesis', 'selection_reasoning',
        'usage', and 'cost' keys
    """
    # Build synthesis options text with labels
    labels = [chr(65 + i) for i in range(len(syntheses))]  # A, B, C, ...
    label_to_model = {label: synth['model'] for label, synth in zip(labels, syntheses)}

    syntheses_text = "\n\n".join([
        f"Synthesis {label}:\n{synth['response']}"
        for label, synth in zip(labels, syntheses)
    ])

    selection_prompt = f"""You are the Supreme Chairman of an LLM Council. Multiple chairman models have each created their own synthesis of the council's responses to a user's question.

Original Question: {user_query}

Here are the different syntheses:

{syntheses_text}

Your task is to:
1. Evaluate each synthesis for:
   - Accuracy and correctness
   - Completeness and comprehensiveness
   - Clarity and readability
   - Proper integration of council insights
2. Select the BEST synthesis
3. Explain your selection reasoning briefly

Format your response as follows:

EVALUATION:
[Brief evaluation of each synthesis]

SELECTED: [Letter of best synthesis, e.g., "A" or "B"]

REASONING:
[Brief explanation of why this synthesis is best]

FINAL RESPONSE:
[Copy the selected synthesis here, or provide an improved version that combines the best elements]"""

    messages = [{"role": "user", "content": selection_prompt}]

    # Query supreme chairman
    supreme_model = get_supreme_chairman_model()
    response = await query_model(supreme_model, messages)

    if response is None:
        # Fallback: return first successful synthesis
        for synth in syntheses:
            if "Error:" not in synth['response']:
                return {
                    "model": supreme_model,
                    "response": synth['response'],
                    "selected_synthesis": synth['model'],
                    "selection_reasoning": "Fallback: Supreme chairman unavailable",
                    "usage": {},
                    "cost": {},
                }
        return {
            "model": supreme_model,
            "response": "Error: Unable to select synthesis.",
            "selected_synthesis": None,
            "selection_reasoning": "All syntheses failed",
            "usage": {},
            "cost": {},
        }

    # Parse the selection response
    content = response.get('content', '')
    selected_model = None
    selection_reasoning = ""
    final_response = content

    # Extract SELECTED synthesis
    if "SELECTED:" in content:
        parts = content.split("SELECTED:")
        if len(parts) >= 2:
            selected_part = parts[1].strip()
            # Extract the letter (first non-whitespace character)
            for char in selected_part:
                if char.isalpha() and char.upper() in label_to_model:
                    selected_model = label_to_model[char.upper()]
                    break

    # Extract REASONING
    if "REASONING:" in content:
        parts = content.split("REASONING:")
        if len(parts) >= 2:
            reasoning_part = parts[1]
            # Get text until FINAL RESPONSE if present
            if "FINAL RESPONSE:" in reasoning_part:
                selection_reasoning = reasoning_part.split("FINAL RESPONSE:")[0].strip()
            else:
                selection_reasoning = reasoning_part.strip()

    # Extract FINAL RESPONSE
    if "FINAL RESPONSE:" in content:
        parts = content.split("FINAL RESPONSE:")
        if len(parts) >= 2:
            final_response = parts[1].strip()

    return {
        "model": supreme_model,
        "response": final_response,
        "selected_synthesis": selected_model,
        "selection_reasoning": selection_reasoning,
        "syntheses": syntheses,  # Include all syntheses for UI display
        "label_to_model": label_to_model,
        "usage": response.get('usage', {}),
        "cost": response.get('cost', {}),
    }


async def stage3_multi_chairman_streaming(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stage 3 Multi-Chairman with streaming support.

    Yields events as syntheses are generated and selection is made.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Yields:
        Events with types:
        - {'type': 'multi_synthesis_start'} - Multi-synthesis begins
        - {'type': 'synthesis_complete', 'model': str, 'response': str, ...} - A chairman finished
        - {'type': 'multi_synthesis_complete', 'syntheses': list} - All chairmen finished
        - {'type': 'selection_start'} - Supreme chairman selection begins
        - {'type': 'selection_token', 'content': str} - Token from supreme chairman
        - {'type': 'selection_complete', 'result': dict} - Selection finished
    """
    # Notify start
    yield {"type": "multi_synthesis_start"}

    # Get all syntheses (batch - we could stream each, but keeping simple)
    syntheses = await stage3_multi_chairman_synthesis(
        user_query, stage1_results, stage2_results
    )

    # Yield each synthesis result
    for synth in syntheses:
        yield {
            "type": "synthesis_complete",
            **synth,
        }

    # Notify all syntheses complete
    yield {
        "type": "multi_synthesis_complete",
        "syntheses": syntheses,
    }

    # Build selection prompt
    labels = [chr(65 + i) for i in range(len(syntheses))]
    label_to_model = {label: synth['model'] for label, synth in zip(labels, syntheses)}

    syntheses_text = "\n\n".join([
        f"Synthesis {label}:\n{synth['response']}"
        for label, synth in zip(labels, syntheses)
    ])

    selection_prompt = f"""You are the Supreme Chairman of an LLM Council. Multiple chairman models have each created their own synthesis of the council's responses to a user's question.

Original Question: {user_query}

Here are the different syntheses:

{syntheses_text}

Your task is to:
1. Evaluate each synthesis for:
   - Accuracy and correctness
   - Completeness and comprehensiveness
   - Clarity and readability
   - Proper integration of council insights
2. Select the BEST synthesis
3. Explain your selection reasoning briefly

Format your response as follows:

EVALUATION:
[Brief evaluation of each synthesis]

SELECTED: [Letter of best synthesis, e.g., "A" or "B"]

REASONING:
[Brief explanation of why this synthesis is best]

FINAL RESPONSE:
[Copy the selected synthesis here, or provide an improved version that combines the best elements]"""

    messages = [{"role": "user", "content": selection_prompt}]

    # Notify selection start
    yield {"type": "selection_start"}

    # Stream supreme chairman selection
    supreme_model = get_supreme_chairman_model()
    accumulated_content = ""

    async for event in query_model_streaming(supreme_model, messages):
        if event["type"] == "token":
            accumulated_content += event["content"]
            yield {
                "type": "selection_token",
                "content": event["content"],
                "model": supreme_model,
            }
        elif event["type"] == "complete":
            # Parse the complete response
            content = accumulated_content or event.get("content", "")
            selected_model = None
            selection_reasoning = ""
            final_response = content

            # Extract SELECTED synthesis
            if "SELECTED:" in content:
                parts = content.split("SELECTED:")
                if len(parts) >= 2:
                    selected_part = parts[1].strip()
                    for char in selected_part:
                        if char.isalpha() and char.upper() in label_to_model:
                            selected_model = label_to_model[char.upper()]
                            break

            # Extract REASONING
            if "REASONING:" in content:
                parts = content.split("REASONING:")
                if len(parts) >= 2:
                    reasoning_part = parts[1]
                    if "FINAL RESPONSE:" in reasoning_part:
                        selection_reasoning = reasoning_part.split("FINAL RESPONSE:")[0].strip()
                    else:
                        selection_reasoning = reasoning_part.strip()

            # Extract FINAL RESPONSE
            if "FINAL RESPONSE:" in content:
                parts = content.split("FINAL RESPONSE:")
                if len(parts) >= 2:
                    final_response = parts[1].strip()

            yield {
                "type": "selection_complete",
                "result": {
                    "model": supreme_model,
                    "response": final_response,
                    "selected_synthesis": selected_model,
                    "selection_reasoning": selection_reasoning,
                    "syntheses": syntheses,
                    "label_to_model": label_to_model,
                    "usage": event.get("usage", {}),
                    "cost": event.get("cost", {}),
                },
            }
        elif event["type"] == "error":
            yield {
                "type": "selection_error",
                "error": event.get("error", "Unknown error"),
            }


def calculate_multi_chairman_costs(
    syntheses: List[Dict[str, Any]],
    selection_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate total costs for multi-chairman synthesis.

    Args:
        syntheses: List of synthesis results from chairmen
        selection_result: Supreme chairman selection result

    Returns:
        Cost breakdown dict with synthesis_costs, selection_cost, and total
    """
    synthesis_costs = []
    for synth in syntheses:
        if synth.get('cost'):
            synthesis_costs.append(synth['cost'])

    selection_cost = selection_result.get('cost', {})

    all_costs = synthesis_costs + ([selection_cost] if selection_cost else [])
    total = aggregate_costs(all_costs)

    return {
        "synthesis_costs": synthesis_costs,
        "selection_cost": selection_cost,
        "total": total,
        "chairman_count": len(syntheses),
    }
