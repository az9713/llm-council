"""
Weighted Consensus Voting for LLM Council.

This module calculates model weights based on historical performance data
from the analytics module. Models that consistently perform well (high win rate,
lower average rank) have their votes weighted more heavily when aggregating rankings.

Weight Calculation Formula:
- Base weight = 1.0 for all models
- Performance bonus = (win_rate / 100) * WIN_RATE_FACTOR
- Rank bonus = (1 / average_rank) * RANK_FACTOR if average_rank > 0
- Final weight = base_weight + performance_bonus + rank_bonus
- Minimum weight = MIN_WEIGHT (prevents outliers from having 0 influence)
- Maximum weight = MAX_WEIGHT (prevents single model dominance)

The weights are normalized so that the average weight across participating models is 1.0,
ensuring the weighted system produces comparable results to unweighted.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import math

from . import analytics

# Weight calculation constants
WIN_RATE_FACTOR = 1.5      # How much win rate affects weight (1.5x multiplier for 100% win rate)
RANK_FACTOR = 0.5          # How much average rank affects weight
MIN_WEIGHT = 0.3           # Minimum weight to prevent complete exclusion
MAX_WEIGHT = 2.5           # Maximum weight to prevent single model dominance
MIN_QUERIES_FOR_WEIGHT = 2 # Minimum queries before historical data affects weights
CONFIDENCE_FACTOR = 0.1    # How much average confidence affects weight


def get_model_weights(
    models: Optional[List[str]] = None,
    include_confidence: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate weights for models based on their historical performance.

    Models with better historical performance (higher win rates, lower average ranks)
    receive higher weights. Models with no history receive the default weight of 1.0.

    Args:
        models: Optional list of specific models to get weights for.
                If None, returns weights for all models with history.
        include_confidence: If True, include confidence scores in weight calculation.

    Returns:
        Dict mapping model names to weight info:
        {
            "model_name": {
                "weight": float,  # The calculated weight (0.3 to 2.5)
                "normalized_weight": float,  # Weight normalized so avg = 1.0
                "win_rate": float,  # Historical win rate percentage
                "average_rank": float,  # Historical average rank
                "total_queries": int,  # Number of queries in history
                "has_history": bool,  # Whether model has enough history
                "weight_explanation": str,  # Human-readable explanation
            }
        }
    """
    # Get model statistics from analytics
    stats = analytics.get_model_statistics()
    model_stats = stats.get("models", {})

    # Calculate weights for each model
    weights = {}

    # Determine which models to process
    models_to_process = models if models else list(model_stats.keys())

    for model in models_to_process:
        model_data = model_stats.get(model, {})
        total_queries = model_data.get("total_queries", 0)

        if total_queries < MIN_QUERIES_FOR_WEIGHT:
            # Not enough history - use default weight
            weights[model] = {
                "weight": 1.0,
                "normalized_weight": 1.0,
                "win_rate": model_data.get("win_rate", 0),
                "average_rank": model_data.get("average_rank"),
                "average_confidence": model_data.get("average_confidence"),
                "total_queries": total_queries,
                "has_history": False,
                "weight_explanation": f"Default weight (need {MIN_QUERIES_FOR_WEIGHT}+ queries for historical weighting)",
            }
        else:
            # Calculate weight based on historical performance
            win_rate = model_data.get("win_rate", 0)  # 0-100
            average_rank = model_data.get("average_rank", 3.0)  # Lower is better
            average_confidence = model_data.get("average_confidence", 5.0)  # 1-10

            # Base weight
            weight = 1.0

            # Add win rate bonus (0 to WIN_RATE_FACTOR)
            win_rate_bonus = (win_rate / 100) * WIN_RATE_FACTOR
            weight += win_rate_bonus

            # Add rank bonus (inverse relationship - lower rank = higher bonus)
            if average_rank and average_rank > 0:
                # Models with avg rank 1.0 get full bonus, rank 3.0 gets 1/3 bonus
                rank_bonus = (1 / average_rank) * RANK_FACTOR
                weight += rank_bonus

            # Add confidence bonus (optional)
            if include_confidence and average_confidence:
                # Normalize confidence (1-10) to 0-1 and apply factor
                confidence_bonus = ((average_confidence - 1) / 9) * CONFIDENCE_FACTOR
                weight += confidence_bonus

            # Clamp to valid range
            weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weight))

            # Build explanation
            explanation_parts = []
            explanation_parts.append(f"Base: 1.0")
            explanation_parts.append(f"Win rate bonus: +{win_rate_bonus:.2f} ({win_rate:.1f}% win rate)")
            if average_rank and average_rank > 0:
                rank_bonus = (1 / average_rank) * RANK_FACTOR
                explanation_parts.append(f"Rank bonus: +{rank_bonus:.2f} (avg rank {average_rank:.2f})")
            if include_confidence and average_confidence:
                confidence_bonus = ((average_confidence - 1) / 9) * CONFIDENCE_FACTOR
                explanation_parts.append(f"Confidence bonus: +{confidence_bonus:.2f} (avg conf {average_confidence:.1f})")

            weights[model] = {
                "weight": round(weight, 3),
                "normalized_weight": round(weight, 3),  # Will be normalized later
                "win_rate": win_rate,
                "average_rank": average_rank,
                "average_confidence": average_confidence,
                "total_queries": total_queries,
                "has_history": True,
                "weight_explanation": " | ".join(explanation_parts),
            }

    # Normalize weights so average is 1.0 (only if we have models with history)
    if weights:
        total_weight = sum(w["weight"] for w in weights.values())
        num_models = len(weights)
        if total_weight > 0 and num_models > 0:
            avg_weight = total_weight / num_models
            if avg_weight > 0:
                for model in weights:
                    weights[model]["normalized_weight"] = round(
                        weights[model]["weight"] / avg_weight, 3
                    )

    return weights


def calculate_weighted_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
    use_weights: bool = True
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models, optionally using historical weights.

    When weights are enabled, models with better historical performance have their
    votes count more when determining the final aggregate ranking.

    Args:
        stage2_results: Rankings from each model (same format as council.py)
        label_to_model: Mapping from anonymous labels to model names
        use_weights: If True, apply historical performance weights to votes

    Returns:
        List of dicts with model name, weighted/unweighted average rank, and weight info,
        sorted best to worst by weighted average rank.
    """
    from .council import parse_ranking_from_text

    # Get the list of models that participated (as responders)
    responder_models = list(label_to_model.values())

    # Get weights for voter models (the models doing the ranking)
    voter_models = [r.get("model") for r in stage2_results if r.get("model")]
    model_weights = get_model_weights(voter_models) if use_weights else {}

    # Track positions for each responding model
    model_positions = defaultdict(list)
    model_weighted_positions = defaultdict(list)
    model_voter_weights = defaultdict(list)

    for ranking in stage2_results:
        voter_model = ranking.get("model")
        ranking_text = ranking.get("ranking", "")

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        # Get this voter's weight
        voter_weight = 1.0
        if use_weights and voter_model in model_weights:
            voter_weight = model_weights[voter_model].get("normalized_weight", 1.0)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)
                model_weighted_positions[model_name].append(position * voter_weight)
                model_voter_weights[model_name].append({
                    "voter": voter_model,
                    "weight": voter_weight,
                    "position": position,
                })

    # Calculate average positions for each model
    aggregate = []
    for model in responder_models:
        positions = model_positions.get(model, [])
        weighted_positions = model_weighted_positions.get(model, [])
        voter_weights_list = model_voter_weights.get(model, [])

        if positions:
            # Unweighted average
            avg_rank = sum(positions) / len(positions)

            # Weighted average: sum(position * weight) / sum(weights)
            total_weight = sum(vw["weight"] for vw in voter_weights_list)
            weighted_avg_rank = (
                sum(vw["position"] * vw["weight"] for vw in voter_weights_list) / total_weight
            ) if total_weight > 0 else avg_rank

            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "weighted_average_rank": round(weighted_avg_rank, 2) if use_weights else round(avg_rank, 2),
                "rankings_count": len(positions),
                "weights_applied": use_weights,
                "rank_change": round(avg_rank - weighted_avg_rank, 2) if use_weights else 0,
            })

    # Sort by weighted average rank (lower is better)
    sort_key = "weighted_average_rank" if use_weights else "average_rank"
    aggregate.sort(key=lambda x: x[sort_key])

    return aggregate


def get_weights_summary() -> Dict[str, Any]:
    """
    Get a summary of all model weights for display purposes.

    Returns:
        Dict with summary information:
        {
            "weights": {...},  # Full weights dict from get_model_weights
            "has_historical_data": bool,  # Whether any models have history
            "models_with_history": int,  # Count of models with enough history
            "total_models_tracked": int,  # Total models in analytics
            "weight_range": {"min": float, "max": float},  # Range of weights
            "explanation": str,  # Human-readable summary
        }
    """
    weights = get_model_weights()

    models_with_history = sum(1 for w in weights.values() if w.get("has_history", False))

    weight_values = [w["weight"] for w in weights.values()] if weights else [1.0]

    has_data = models_with_history > 0

    explanation = (
        f"Weights based on performance across {models_with_history} models with sufficient history. "
        f"Weight range: {min(weight_values):.2f} to {max(weight_values):.2f}. "
        "Higher weights = more influence on aggregate rankings."
    ) if has_data else (
        f"No historical data yet (need {MIN_QUERIES_FOR_WEIGHT}+ queries per model). "
        "All models have equal weight of 1.0."
    )

    return {
        "weights": weights,
        "has_historical_data": has_data,
        "models_with_history": models_with_history,
        "total_models_tracked": len(weights),
        "weight_range": {
            "min": round(min(weight_values), 3),
            "max": round(max(weight_values), 3),
        },
        "explanation": explanation,
    }
