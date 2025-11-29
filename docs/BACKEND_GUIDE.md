# Backend Guide

This guide provides a detailed explanation of every backend file, function, and line of code. It's designed for developers with no prior LLM application experience.

## Table of Contents

1. [Overview](#overview)
2. [config.py - Default Configuration](#configpy---default-configuration)
3. [config_api.py - Dynamic Configuration](#config_apipy---dynamic-configuration)
4. [pricing.py - Cost Calculation](#pricingpy---cost-calculation)
5. [openrouter.py - API Client](#openrouterpy---api-client)
6. [council.py - Core Logic](#councilpy---core-logic)
7. [storage.py - Data Persistence](#storagepy---data-persistence)
8. [main.py - HTTP API](#mainpy---http-api)
9. [Running the Backend](#running-the-backend)

---

## Overview

The backend is a Python application using:
- **FastAPI**: Modern web framework for building APIs
- **httpx**: Async HTTP client for API calls
- **Pydantic**: Data validation
- **python-dotenv**: Environment variable loading

Files:
```
backend/
├── __init__.py      # Package marker (empty)
├── config.py        # Default configuration values
├── config_api.py    # Dynamic configuration management
├── pricing.py       # Model pricing and cost calculation (NEW)
├── openrouter.py    # OpenRouter API client
├── council.py       # 3-stage deliberation logic
├── storage.py       # JSON file storage
└── main.py          # FastAPI application
```

---

## config.py - Default Configuration

**Location**: `backend/config.py`

This file contains default configuration values and environment variable loading. Note that model configuration (council models and chairman) is now managed dynamically by `config_api.py` - these defaults are only used when no configuration file exists.

### Complete Code with Explanations

```python
"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads the .env file in the project root and makes its contents
# available via os.getenv()
load_dotenv()

# OpenRouter API key
# This is read from the .env file. os.getenv() returns None if not found.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
# These are the models that will answer questions and rank each other
# Format: "provider/model-name"
COUNCIL_MODELS = [
    "openai/gpt-5.1",               # OpenAI's GPT-5.1
    "google/gemini-3-pro-preview",  # Google's Gemini 3 Pro
    "anthropic/claude-sonnet-4.5",  # Anthropic's Claude Sonnet
    "x-ai/grok-4",                  # xAI's Grok 4
]

# Chairman model - synthesizes final response
# Can be any model, often one of the council members
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# OpenRouter API endpoint
# This is the standard OpenRouter chat completions endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
# Relative path from project root
DATA_DIR = "data/conversations"
```

### Key Concepts

**Environment Variables**: Sensitive data (like API keys) should not be in code. The `.env` file keeps secrets separate and is not committed to git.

**Model Identifiers**: OpenRouter uses `provider/model-name` format. You can find available models at [openrouter.ai/models](https://openrouter.ai/models).

---

## config_api.py - Dynamic Configuration

**Location**: `backend/config_api.py`

This file manages dynamic model configuration that can be changed via the UI without editing code. Configuration is stored in `data/council_config.json`.

### Complete Code with Explanations

```python
"""Dynamic configuration management for LLM Council.

This module handles loading, saving, and validating model configuration.
Configuration is stored in a JSON file and can be modified via API endpoints.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configuration file path (relative to project root)
CONFIG_FILE = "data/council_config.json"

# Default configuration (used when no config file exists)
DEFAULT_CONFIG = {
    "council_models": [
        "openai/gpt-5.1",
        "google/gemini-3-pro-preview",
        "anthropic/claude-sonnet-4.5",
        "x-ai/grok-4",
    ],
    "chairman_model": "google/gemini-3-pro-preview",
}

# Minimum number of council models required
MIN_COUNCIL_MODELS = 2


def _ensure_config_dir() -> None:
    """Ensure the data directory exists."""
    config_path = Path(CONFIG_FILE)
    config_path.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """
    Load configuration from the JSON file.
    If the config file doesn't exist, returns the default configuration.

    Returns:
        Dict containing council_models and chairman_model
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                return {
                    "council_models": config.get("council_models", DEFAULT_CONFIG["council_models"]),
                    "chairman_model": config.get("chairman_model", DEFAULT_CONFIG["chairman_model"]),
                }
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load config file, using defaults: {e}")
            return DEFAULT_CONFIG.copy()

    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save configuration to the JSON file.

    Args:
        config: Dict containing council_models and chairman_model

    Returns:
        The saved configuration

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate before saving
    validation_error = validate_config(config)
    if validation_error:
        raise ValueError(validation_error)

    _ensure_config_dir()

    config_to_save = {
        "council_models": config["council_models"],
        "chairman_model": config["chairman_model"],
    }

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=2)

    return config_to_save


def validate_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Validate configuration.

    Args:
        config: Configuration dict to validate

    Returns:
        Error message if invalid, None if valid
    """
    # Check council_models exists and is a list
    if "council_models" not in config:
        return "council_models is required"

    if not isinstance(config["council_models"], list):
        return "council_models must be a list"

    # Check minimum number of models
    if len(config["council_models"]) < MIN_COUNCIL_MODELS:
        return f"At least {MIN_COUNCIL_MODELS} council models are required"

    # Check all models are non-empty strings
    for i, model in enumerate(config["council_models"]):
        if not isinstance(model, str) or not model.strip():
            return f"council_models[{i}] must be a non-empty string"

    # Check chairman_model exists and is a non-empty string
    if "chairman_model" not in config:
        return "chairman_model is required"

    if not isinstance(config["chairman_model"], str) or not config["chairman_model"].strip():
        return "chairman_model must be a non-empty string"

    return None


def get_council_models() -> List[str]:
    """Get the current list of council models."""
    config = load_config()
    return config["council_models"]


def get_chairman_model() -> str:
    """Get the current chairman model."""
    config = load_config()
    return config["chairman_model"]


def reset_to_defaults() -> Dict[str, Any]:
    """Reset configuration to defaults."""
    return save_config(DEFAULT_CONFIG.copy())


# List of popular models for the frontend dropdown suggestions
AVAILABLE_MODELS = [
    # OpenAI
    "openai/gpt-5.1",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/o3",
    "openai/o4-mini",
    # Anthropic
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-haiku-3.5",
    # Google
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    # xAI
    "x-ai/grok-4",
    "x-ai/grok-3",
    # Meta
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    # DeepSeek
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat",
    # Mistral
    "mistralai/mistral-large",
    "mistralai/mistral-medium",
]
```

### Key Concepts

**Runtime Configuration**: Unlike `config.py` which is static, `config_api.py` allows changes without restarting the server. The config is read fresh on each request.

**JSON File Storage**: Configuration is stored in `data/council_config.json` alongside conversations. This keeps all user data in one place.

**Validation**: The `validate_config()` function enforces business rules:
- At least 2 council models (required for meaningful ranking)
- All model identifiers must be non-empty strings
- Chairman model must be specified

**Graceful Defaults**: If the config file doesn't exist or is corrupted, the system falls back to defaults. This ensures the app always works.

### How council.py Uses Dynamic Config

The council orchestration functions now call `get_council_models()` and `get_chairman_model()` instead of importing static values:

```python
# In council.py (before - static):
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL
responses = await query_models_parallel(COUNCIL_MODELS, messages)

# In council.py (after - dynamic):
from .config_api import get_council_models, get_chairman_model
council_models = get_council_models()
responses = await query_models_parallel(council_models, messages)
```

This means configuration changes take effect immediately on the next query.

---

## pricing.py - Cost Calculation

**Location**: `backend/pricing.py`

This file handles model pricing data and cost calculations for tracking query costs.

### Key Components

```python
# Pricing per 1M tokens (input/output)
# Format: "provider/model": {"input": price_per_1M, "output": price_per_1M}
MODEL_PRICING = {
    "openai/gpt-5.1": {"input": 2.50, "output": 10.00},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "google/gemini-3-pro-preview": {"input": 1.25, "output": 5.00},
    # ... more models
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 5.00, "output": 15.00}
```

### Functions

```python
def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a model, with fallback to defaults."""
    return MODEL_PRICING.get(model, DEFAULT_PRICING)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
    """
    Calculate cost breakdown for a model query.

    Returns:
        - input_cost: Cost for prompt tokens (USD)
        - output_cost: Cost for completion tokens (USD)
        - total_cost: Total cost (USD)
        - prompt_tokens, completion_tokens, total_tokens
        - pricing: The rates used
    """
    pricing = get_model_pricing(model)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    # ...


def format_cost(cost: float) -> str:
    """Format cost for display (e.g., '$0.0012' or '<$0.0001')."""


def aggregate_costs(cost_list: list) -> Dict[str, Any]:
    """
    Aggregate multiple cost entries into a summary.
    Used to sum costs across models in a stage or across all stages.
    """
```

### Key Concepts

**Per-Million Pricing**: OpenRouter (and most LLM APIs) price by tokens per million. The formula is:
```python
cost = (tokens / 1_000_000) * price_per_million
```

**Input vs Output Pricing**: Input (prompt) tokens are typically cheaper than output (completion) tokens because generating text is more computationally expensive.

**Default Fallback**: Unknown models use conservative default pricing to avoid underreporting costs.

---

## openrouter.py - API Client

**Location**: `backend/openrouter.py`

This file handles communication with the OpenRouter API.

### Complete Code with Explanations

```python
"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details',
        or None if failed
    """
    # HTTP headers for authentication
    # Authorization uses "Bearer" token format
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Request body follows OpenAI's chat completions format
    # OpenRouter maintains compatibility with this format
    payload = {
        "model": model,      # Which model to use
        "messages": messages, # Conversation history
    }

    try:
        # httpx.AsyncClient is an async HTTP client
        # 'async with' ensures the client is properly closed after use
        async with httpx.AsyncClient(timeout=timeout) as client:
            # POST request to the API
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload  # Automatically serializes to JSON
            )
            # Raises exception if status code indicates error (4xx, 5xx)
            response.raise_for_status()

            # Parse JSON response
            data = response.json()
            # OpenRouter returns responses in this structure:
            # {
            #   "choices": [
            #     {
            #       "message": {
            #         "content": "The response text...",
            #         "reasoning_details": {...}  // Optional, for reasoning models
            #       }
            #     }
            #   ]
            # }
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        # Log the error but don't crash
        # This enables graceful degradation - other models can still work
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create a list of coroutines (async function calls)
    # Each query_model() call is NOT executed yet - just prepared
    tasks = [query_model(model, messages) for model in models]

    # asyncio.gather() runs all tasks concurrently
    # This is MUCH faster than running them one by one
    # If we have 4 models and each takes 5 seconds:
    #   - Sequential: 4 × 5 = 20 seconds
    #   - Parallel: ~5 seconds (all run simultaneously)
    responses = await asyncio.gather(*tasks)

    # Combine models with their responses into a dictionary
    # zip() pairs up corresponding items from both lists
    return {model: response for model, response in zip(models, responses)}
```

### Key Concepts

**Async/Await**: Python's way of handling concurrent operations. `async def` defines an async function, `await` pauses execution until the operation completes.

**httpx**: Similar to `requests` but supports async operations. Essential for non-blocking I/O.

**asyncio.gather()**: Runs multiple async operations concurrently. Critical for performance.

**Error Handling**: Returns `None` on failure instead of raising exceptions. This allows the system to continue even if one model fails.

---

## council.py - Core Logic

**Location**: `backend/council.py`

This is the heart of the application - the 3-stage deliberation logic.

### Complete Code with Explanations

```python
"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple
from .openrouter import query_models_parallel, query_model
from .config_api import get_council_models, get_chairman_model  # Dynamic config


async def stage1_collect_responses(
    user_query: str,
    system_prompt: str = None
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        system_prompt: Optional system prompt to prepend to all queries

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    # Format the query as a chat message
    # OpenRouter/OpenAI format expects a list of messages
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_query})

    # Get current council models dynamically and query all simultaneously
    council_models = get_council_models()
    responses = await query_models_parallel(council_models, messages)

    # Format results, including reasoning_details for reasoning models
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            result = {
                "model": model,
                "response": response.get('content', '')
            }
            # Include reasoning_details if present (for reasoning models like o1, o3)
            # This allows the frontend to display the model's thinking process
            reasoning = response.get('reasoning_details')
            if reasoning:
                result["reasoning_details"] = reasoning
            stage1_results.append(result)

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels: A, B, C, D...
    # chr(65) = 'A', chr(66) = 'B', etc.
    labels = [chr(65 + i) for i in range(len(stage1_results))]

    # Create mapping from label to actual model name
    # This lets us "decode" the anonymization later
    # Example: {"Response A": "openai/gpt-5.1", "Response B": "anthropic/claude-sonnet-4.5"}
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the anonymized responses text
    # Example output:
    # Response A:
    # {GPT's answer}
    #
    # Response B:
    # {Claude's answer}
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    # The ranking prompt is carefully designed to:
    # 1. Provide context (the original question)
    # 2. Show all responses (anonymized)
    # 3. Request evaluation with specific format
    # 4. Ensure parseable output
    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel (dynamic config)
    council_models = get_council_models()
    responses = await query_models_parallel(council_models, messages)

    # Format results with parsed rankings
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            # Parse the ranking from the text
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,        # Raw evaluation text
                "parsed_ranking": parsed     # Extracted ranking list
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build context from Stage 1 responses
    # This time we include the actual model names
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    # Build context from Stage 2 rankings
    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    # Chairman prompt provides full context for synthesis
    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

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

    # Query the chairman model (dynamic config)
    chairman_model = get_chairman_model()
    response = await query_model(chairman_model, messages)

    if response is None:
        # Fallback error message if chairman fails
        return {
            "model": chairman_model,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": chairman_model,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
        Example: ["Response C", "Response A", "Response B", "Response D"]
    """
    import re

    # Primary method: Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Split on the marker and take everything after
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]

            # Try to extract numbered list format: "1. Response A"
            # Regex breakdown:
            #   \d+     - One or more digits (the number)
            #   \.      - Literal period
            #   \s*     - Zero or more whitespace
            #   Response [A-Z] - "Response" followed by single capital letter
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)

            if numbered_matches:
                # Extract just "Response X" from each match
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Last resort fallback: find any "Response X" in the entire text
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track all positions for each model
    # defaultdict(list) creates empty list for new keys automatically
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']
        parsed_ranking = parse_ranking_from_text(ranking_text)

        # For each position in the ranking (1st, 2nd, 3rd, etc.)
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                # Record this model got position X in this ranking
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            # Average of all positions (lower is better)
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)  # How many models ranked this one
            })

    # Sort by average rank (lower is better, so ascending order)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use a fast, cheap model for title generation
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        return "New Conversation"  # Fallback

    title = response.get('content', 'New Conversation').strip()

    # Clean up: remove quotes
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    system_prompt: str = None
) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question
        system_prompt: Optional system prompt to prepend to all queries

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query, system_prompt)

    # Error handling: all models failed
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2: Collect rankings (returns tuple)
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Chairman synthesis
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )

    # Package metadata for frontend
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage3_result, metadata
```

### Key Concepts

**Type Hints**: `List[Dict[str, Any]]` indicates "a list of dictionaries with string keys and any value type". Helps with code readability and IDE support.

**Tuple Returns**: Functions like `stage2_collect_rankings` return multiple values as a tuple, which Python can unpack: `results, mapping = await stage2...`

**Regex Parsing**: `re.findall()` extracts all occurrences matching a pattern. Critical for extracting rankings from free-form text.

**defaultdict**: A dictionary that automatically creates default values for missing keys. `defaultdict(list)` creates empty lists.

### Title Generation Feature

The `generate_conversation_title()` function automatically creates a short title for new conversations:

```python
# Uses a fast, cheap model (hardcoded, not configurable via config.py)
response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)
```

**Important notes**:
- Uses `google/gemini-2.5-flash` (hardcoded) for speed and low cost
- Runs in **parallel** with the 3-stage process (doesn't block)
- Falls back to "New Conversation" if generation fails
- Truncates titles longer than 50 characters
- Strips quotes from generated titles

**To customize the title model**, edit `backend/council.py` line 278.

### Reasoning Model Support

Some models (like OpenAI's o1, o3, or DeepSeek-R1) expose their internal reasoning process via a `reasoning_details` field in the API response. The council captures and passes this through:

```python
# In stage1_collect_responses:
reasoning = response.get('reasoning_details')
if reasoning:
    result["reasoning_details"] = reasoning
```

**How it works**:
1. `openrouter.py` already captures `reasoning_details` from the API response
2. `stage1_collect_responses()` includes it in results when present
3. The frontend displays it in a collapsible "Show Thinking Process" section

**Supported formats** (handled by frontend):
- String: Direct thinking text
- Object with `content` property: `{"type": "thinking", "content": "..."}`
- Array of steps: `[{"content": "step 1"}, {"content": "step 2"}]`

**Note**: Not all models return reasoning details. The field is only included when present, keeping responses lean for non-reasoning models.

### Ranking Parser Fallback Behavior

The `parse_ranking_from_text()` function has three fallback levels:

1. **Primary**: Looks for `FINAL RANKING:` header, extracts numbered list
2. **Secondary**: If no numbered list, extracts any `Response [A-Z]` patterns after the header
3. **Fallback**: If no header found, extracts all `Response [A-Z]` patterns from entire text

This ensures rankings are extracted even when models don't follow the exact format.

---

## storage.py - Data Persistence

**Location**: `backend/storage.py`

This file handles saving and loading conversations from JSON files.

### Complete Code with Explanations

```python
"""JSON-based storage for conversations."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR


def ensure_data_dir():
    """Ensure the data directory exists."""
    # Path.mkdir with parents=True creates parent directories too
    # exist_ok=True means don't error if it already exists
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    # os.path.join handles path separators correctly across OS
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        New conversation dict
    """
    ensure_data_dir()

    # Initial conversation structure
    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),  # ISO 8601 format
        "title": "New Conversation",
        "tags": [],  # Array of tag strings for categorization
        "messages": []  # Will contain user and assistant messages
    }

    # Save to file immediately
    path = get_conversation_path(conversation_id)
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)  # indent=2 for readability

    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    path = get_conversation_path(conversation_id)

    # Check if file exists before trying to read
    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        return json.load(f)


def save_conversation(conversation: Dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    ensure_data_dir()

    path = get_conversation_path(conversation['id'])
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    conversations = []
    # Iterate through all JSON files in the data directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                # Return only metadata, not full message content
                conversations.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "title": data.get("title", "New Conversation"),
                    "tags": data.get("tags", []),  # Include tags in metadata
                    "message_count": len(data["messages"])
                })

    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)

    return conversations


def add_user_message(conversation_id: str, content: str):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    # Append the user message
    conversation["messages"].append({
        "role": "user",
        "content": content
    })

    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any]
):
    """
    Add an assistant message with all 3 stages to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses
        stage2: List of model rankings
        stage3: Final synthesized response
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    # Assistant messages have a different structure than user messages
    # They contain all three stages
    conversation["messages"].append({
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3
    })

    save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["title"] = title
    save_conversation(conversation)


def update_conversation_tags(conversation_id: str, tags: List[str]) -> Optional[Dict[str, Any]]:
    """
    Update tags for a conversation.

    Args:
        conversation_id: Conversation identifier
        tags: New list of tags to set

    Returns:
        Updated conversation dict, or None if not found
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return None

    conversation["tags"] = tags
    save_conversation(conversation)
    return conversation


def get_all_tags() -> List[str]:
    """
    Get all unique tags across all conversations.

    Returns:
        Sorted list of unique tag strings
    """
    ensure_data_dir()
    tags = set()  # Use set to avoid duplicates

    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                # Add all tags from this conversation to the set
                tags.update(data.get("tags", []))

    return sorted(list(tags))  # Return sorted list


def filter_conversations_by_tag(tag: str) -> List[Dict[str, Any]]:
    """
    Get conversations that have a specific tag.

    Args:
        tag: The tag to filter by

    Returns:
        List of conversation metadata dicts that have the tag
    """
    all_convs = list_conversations()
    return [conv for conv in all_convs if tag in conv.get("tags", [])]
```

### Key Concepts

**JSON Storage**: Simple, human-readable, no database required. Good for prototypes but has limitations (no concurrent write safety, all-or-nothing reads).

**Path Handling**: `os.path.join()` creates paths correctly regardless of operating system (Windows uses `\`, Unix uses `/`).

**Error Handling**: Functions return `None` for missing conversations rather than raising exceptions, letting callers decide how to handle it.

---

## main.py - HTTP API

**Location**: `backend/main.py`

This file defines the FastAPI application and HTTP endpoints.

### Complete Code with Explanations

```python
"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio

from . import storage
from . import config_api  # Dynamic configuration
from .council import (
    run_full_council,
    generate_conversation_title,
    stage1_collect_responses,
    stage2_collect_rankings,
    stage3_synthesize_final,
    calculate_aggregate_rankings
)

# Create FastAPI application instance
app = FastAPI(title="LLM Council API")

# Enable CORS (Cross-Origin Resource Sharing)
# This allows the frontend (different origin/port) to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Allowed frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Pydantic models for request/response validation
# These define the expected shape of data

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    pass  # Empty - no data needed to create a conversation


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str  # Required field
    system_prompt: str = None  # Optional system prompt for all models


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    tags: List[str] = []  # Tags for categorization
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    tags: List[str] = []  # Tags for categorization
    messages: List[Dict[str, Any]]


class UpdateTagsRequest(BaseModel):
    """Request to update tags for a conversation."""
    tags: List[str]


class UpdateConfigRequest(BaseModel):
    """Request to update model configuration."""
    council_models: List[str]  # List of model identifiers
    chairman_model: str        # Chairman model identifier


class ConfigResponse(BaseModel):
    """Response containing model configuration."""
    council_models: List[str]
    chairman_model: str


class AvailableModelsResponse(BaseModel):
    """Response containing list of suggested models."""
    models: List[str]


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations(tag: Optional[str] = Query(None, description="Filter by tag")):
    """List all conversations (metadata only), optionally filtered by tag."""
    if tag:
        return storage.filter_conversations_by_tag(tag)
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    # Generate unique ID using UUID
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        # Return 404 Not Found if conversation doesn't exist
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process.
    Returns the complete response with all stages (non-streaming).
    """
    # Verify conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Check if this is the first message (for title generation)
    is_first_message = len(conversation["messages"]) == 0

    # Save user message
    storage.add_user_message(conversation_id, request.content)

    # Generate title if first message
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Run the full 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
        request.content,
        system_prompt=request.system_prompt
    )

    # Save assistant response
    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result
    )

    # Return everything to the client
    return {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the 3-stage council process.
    Returns Server-Sent Events as each stage completes.
    """
    # Verify conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    is_first_message = len(conversation["messages"]) == 0

    async def event_generator():
        """
        Generator function that yields SSE events.
        Each yield sends data to the client immediately.
        """
        try:
            # Save user message
            storage.add_user_message(conversation_id, request.content)

            # Start title generation in parallel (don't wait for it)
            title_task = None
            if is_first_message:
                # asyncio.create_task() starts the coroutine immediately
                # but doesn't wait for it to complete
                title_task = asyncio.create_task(
                    generate_conversation_title(request.content)
                )

            # Stage 1
            # SSE format: "data: {json}\n\n"
            yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
            stage1_results = await stage1_collect_responses(request.content, request.system_prompt)
            yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

            # Stage 2
            yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
            stage2_results, label_to_model = await stage2_collect_rankings(
                request.content, stage1_results
            )
            aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
            yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

            # Stage 3
            yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
            stage3_result = await stage3_synthesize_final(
                request.content, stage1_results, stage2_results
            )
            yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

            # Wait for title if we started generating one
            if title_task:
                title = await title_task
                storage.update_conversation_title(conversation_id, title)
                yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

            # Save complete assistant message
            storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result
            )

            # Signal completion
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            # Send error to client
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    # Return streaming response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",  # SSE content type
        headers={
            "Cache-Control": "no-cache",  # Don't cache the stream
            "Connection": "keep-alive",   # Keep connection open
        }
    )


@app.put("/api/conversations/{conversation_id}/tags")
async def update_tags(conversation_id: str, request: UpdateTagsRequest):
    """Update tags for a conversation."""
    conversation = storage.update_conversation_tags(conversation_id, request.tags)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok", "tags": request.tags}


@app.get("/api/tags")
async def get_all_tags():
    """Get all unique tags across all conversations."""
    return {"tags": storage.get_all_tags()}


# Configuration Endpoints

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current model configuration."""
    return config_api.load_config()


@app.put("/api/config", response_model=ConfigResponse)
async def update_config(request: UpdateConfigRequest):
    """
    Update model configuration.
    Validates that at least 2 council models are specified.
    Configuration is persisted to disk.
    """
    try:
        config = config_api.save_config({
            "council_models": request.council_models,
            "chairman_model": request.chairman_model,
        })
        return config
    except ValueError as e:
        # Validation failed
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/config/reset", response_model=ConfigResponse)
async def reset_config():
    """Reset configuration to defaults."""
    return config_api.reset_to_defaults()


@app.get("/api/config/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get list of suggested models for the dropdown."""
    return {"models": config_api.AVAILABLE_MODELS}


# Entry point when running directly
if __name__ == "__main__":
    import uvicorn
    # Start the server on all interfaces (0.0.0.0) port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Key Concepts

**FastAPI Decorators**: `@app.get("/path")` and `@app.post("/path")` define HTTP endpoints. The function below handles requests to that path.

**Pydantic Models**: Define expected request/response shapes. FastAPI automatically validates incoming data.

**HTTPException**: Raises HTTP errors with proper status codes (404, 400, etc.).

**Server-Sent Events (SSE)**: A way to stream data from server to client. The generator yields events that are sent immediately.

**StreamingResponse**: FastAPI response type that streams content as it's generated instead of waiting for everything to complete.

---

## Running the Backend

### From Project Root

```bash
# Using uv (recommended)
uv run python -m backend.main

# Or without uv (if dependencies installed globally)
python -m backend.main
```

### Important Notes

1. **Always use `-m` flag**: The `-m` tells Python to run `backend.main` as a module, which is required for relative imports to work.

2. **Run from project root**: Not from inside the `backend/` directory.

3. **Port 8001**: The backend runs on port 8001, not the common default of 8000.

### Verifying It's Running

Open http://localhost:8001 in your browser. You should see:

```json
{"status": "ok", "service": "LLM Council API"}
```

---

## Next Steps

- **Understand the frontend**: [Frontend Guide](./FRONTEND_GUIDE.md)
- **See all API endpoints**: [API Reference](./API_REFERENCE.md)
- **Add new features**: [Extending the Codebase](./EXTENDING.md)
