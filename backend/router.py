"""Dynamic model routing based on question type classification.

This module provides intelligent model selection by classifying questions
into categories and routing them to model pools optimized for each type.

Categories:
- CODING: Programming, debugging, code review, technical implementation
- CREATIVE: Writing, brainstorming, storytelling, poetry, humor
- FACTUAL: Research, facts, historical information, definitions
- ANALYSIS: Logical reasoning, comparisons, evaluations, strategies
- GENERAL: Broad questions, advice, multi-domain queries (uses all models)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .openrouter import query_model


class QuestionCategory(str, Enum):
    """Categories for question classification."""
    CODING = "coding"
    CREATIVE = "creative"
    FACTUAL = "factual"
    ANALYSIS = "analysis"
    GENERAL = "general"


# Default model pools optimized for each question type
# These can be overridden via configuration
DEFAULT_MODEL_POOLS = {
    QuestionCategory.CODING: [
        "anthropic/claude-sonnet-4.5",  # Excellent at code
        "openai/gpt-5.1",               # Strong coding
        "deepseek/deepseek-chat",       # Good at code
        "openai/gpt-4.1",               # Reliable coding
    ],
    QuestionCategory.CREATIVE: [
        "anthropic/claude-sonnet-4.5",  # Creative and eloquent
        "google/gemini-3-pro-preview",  # Strong creative
        "openai/gpt-5.1",               # Creative capabilities
        "x-ai/grok-4",                  # Unique perspectives
    ],
    QuestionCategory.FACTUAL: [
        "google/gemini-3-pro-preview",  # Large knowledge base
        "openai/gpt-5.1",               # Strong factual recall
        "anthropic/claude-sonnet-4.5",  # Accurate information
        "openai/gpt-4.1",               # Reliable facts
    ],
    QuestionCategory.ANALYSIS: [
        "openai/gpt-5.1",               # Strong reasoning
        "anthropic/claude-sonnet-4.5",  # Analytical thinking
        "openai/o3",                    # Reasoning model
        "google/gemini-3-pro-preview",  # Balanced analysis
    ],
    QuestionCategory.GENERAL: None,  # Uses all configured council models
}


# Fast classification model (cheap and quick)
CLASSIFIER_MODEL = "google/gemini-2.5-flash"


# Keywords for quick heuristic classification (fallback if LLM fails)
CATEGORY_KEYWORDS = {
    QuestionCategory.CODING: [
        "code", "program", "function", "class", "debug", "error", "bug",
        "python", "javascript", "typescript", "java", "c++", "rust", "go",
        "api", "database", "sql", "git", "deploy", "docker", "algorithm",
        "data structure", "compile", "runtime", "syntax", "variable",
        "implement", "refactor", "test", "unit test", "regex", "html", "css",
    ],
    QuestionCategory.CREATIVE: [
        "write", "story", "poem", "creative", "imagine", "fiction", "novel",
        "character", "plot", "dialogue", "brainstorm", "idea", "humor",
        "joke", "funny", "song", "lyrics", "script", "screenplay", "essay",
        "blog", "article", "slogan", "tagline", "marketing", "brand",
    ],
    QuestionCategory.FACTUAL: [
        "what is", "who is", "when did", "where is", "how many", "define",
        "explain", "history", "fact", "date", "year", "country", "capital",
        "population", "science", "physics", "chemistry", "biology", "math",
        "formula", "theorem", "law", "discovery", "invention", "founded",
    ],
    QuestionCategory.ANALYSIS: [
        "compare", "contrast", "analyze", "evaluate", "pros and cons",
        "advantage", "disadvantage", "best", "worst", "strategy", "plan",
        "decision", "choose", "recommend", "should i", "trade-off",
        "reasoning", "logic", "argument", "critique", "review", "assess",
    ],
}


def classify_by_keywords(query: str) -> QuestionCategory:
    """
    Quick heuristic classification based on keyword matching.

    Used as a fallback when LLM classification fails.

    Args:
        query: The user's question

    Returns:
        QuestionCategory based on keyword matching, or GENERAL if no match
    """
    query_lower = query.lower()

    # Count keyword matches for each category
    scores = {category: 0 for category in QuestionCategory}

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                scores[category] += 1

    # Find category with highest score
    max_score = max(scores.values())

    if max_score == 0:
        return QuestionCategory.GENERAL

    # Return the category with the highest score
    for category, score in scores.items():
        if score == max_score:
            return category

    return QuestionCategory.GENERAL


async def classify_question(query: str) -> Tuple[QuestionCategory, float, str]:
    """
    Classify a question into a category using an LLM.

    Uses a fast, cheap model to classify the question type, enabling
    intelligent routing to specialized model pools.

    Args:
        query: The user's question

    Returns:
        Tuple of (category, confidence, reasoning):
        - category: The detected QuestionCategory
        - confidence: Confidence score 0.0-1.0
        - reasoning: Brief explanation of classification
    """
    classification_prompt = f"""Classify the following question into exactly ONE of these categories:

CODING - Questions about programming, debugging, code review, algorithms, technical implementation
CREATIVE - Requests for creative writing, brainstorming, storytelling, poetry, humor
FACTUAL - Questions seeking facts, definitions, historical information, scientific data
ANALYSIS - Questions requiring comparison, evaluation, logical reasoning, strategic thinking
GENERAL - Broad questions, advice, or queries that span multiple domains

Question: {query}

Respond in EXACTLY this format (no other text):
CATEGORY: [category name in caps]
CONFIDENCE: [number 0.0 to 1.0]
REASONING: [one sentence explanation]"""

    messages = [{"role": "user", "content": classification_prompt}]

    try:
        response = await query_model(CLASSIFIER_MODEL, messages, timeout=15.0)

        if response is None:
            # Fallback to keyword classification
            category = classify_by_keywords(query)
            return category, 0.5, "Fallback: LLM classification failed"

        content = response.get("content", "")

        # Parse the response
        category = QuestionCategory.GENERAL
        confidence = 0.7
        reasoning = "Parsed from LLM response"

        # Extract category
        category_match = re.search(r'CATEGORY:\s*(\w+)', content, re.IGNORECASE)
        if category_match:
            category_str = category_match.group(1).upper()
            try:
                category = QuestionCategory(category_str.lower())
            except ValueError:
                # Invalid category, use keyword fallback
                category = classify_by_keywords(query)
                reasoning = f"Invalid category '{category_str}', used keyword fallback"

        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            except ValueError:
                confidence = 0.7

        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', content)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return category, confidence, reasoning

    except Exception as e:
        # Fallback to keyword classification on any error
        category = classify_by_keywords(query)
        return category, 0.5, f"Fallback: {str(e)}"


def get_model_pool(
    category: QuestionCategory,
    all_models: List[str],
    custom_pools: Optional[Dict[str, List[str]]] = None
) -> List[str]:
    """
    Get the appropriate model pool for a question category.

    Args:
        category: The question category
        all_models: All available council models (used for GENERAL category)
        custom_pools: Optional custom pool configuration

    Returns:
        List of model identifiers to use for this query
    """
    # Use custom pools if provided, otherwise use defaults
    pools = custom_pools or DEFAULT_MODEL_POOLS

    # GENERAL category always uses all configured models
    if category == QuestionCategory.GENERAL:
        return all_models

    # Get pool for this category
    pool = pools.get(category)

    if pool is None:
        return all_models

    # Filter pool to only include models that are available
    # (intersection of pool and all_models preserves pool order)
    available_pool = [model for model in pool if model in all_models]

    # If no models from the pool are available, use all models
    if not available_pool:
        return all_models

    # Ensure we have at least 2 models (minimum for council)
    if len(available_pool) < 2:
        # Add models from all_models that aren't in the pool
        for model in all_models:
            if model not in available_pool:
                available_pool.append(model)
            if len(available_pool) >= 2:
                break

    return available_pool


async def route_query(
    query: str,
    all_models: List[str],
    custom_pools: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Route a query to the appropriate model pool.

    Main entry point for dynamic routing. Classifies the question
    and returns routing information.

    Args:
        query: The user's question
        all_models: All configured council models
        custom_pools: Optional custom pool configuration

    Returns:
        Dict with:
        - category: QuestionCategory (string value)
        - confidence: Classification confidence (0.0-1.0)
        - reasoning: Brief explanation
        - models: List of models to use
        - is_routed: Whether routing was applied (False if GENERAL)
        - original_model_count: Number of models before routing
        - routed_model_count: Number of models after routing
    """
    # Classify the question
    category, confidence, reasoning = await classify_question(query)

    # Get appropriate model pool
    models = get_model_pool(category, all_models, custom_pools)

    return {
        "category": category.value,
        "confidence": confidence,
        "reasoning": reasoning,
        "models": models,
        "is_routed": category != QuestionCategory.GENERAL,
        "original_model_count": len(all_models),
        "routed_model_count": len(models),
    }


# Mapping of category to display name and description
CATEGORY_INFO = {
    QuestionCategory.CODING: {
        "name": "Coding",
        "description": "Programming, debugging, and technical implementation",
        "color": "#22c55e",  # Green
    },
    QuestionCategory.CREATIVE: {
        "name": "Creative",
        "description": "Writing, brainstorming, and creative tasks",
        "color": "#a855f7",  # Purple
    },
    QuestionCategory.FACTUAL: {
        "name": "Factual",
        "description": "Facts, definitions, and information lookup",
        "color": "#3b82f6",  # Blue
    },
    QuestionCategory.ANALYSIS: {
        "name": "Analysis",
        "description": "Comparison, evaluation, and reasoning",
        "color": "#f59e0b",  # Amber
    },
    QuestionCategory.GENERAL: {
        "name": "General",
        "description": "Broad questions using all models",
        "color": "#6b7280",  # Gray
    },
}
