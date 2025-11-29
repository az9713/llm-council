"""Confidence-Gated Escalation for LLM Council.

This module implements tiered model selection based on confidence and agreement levels.
Queries start with cheaper Tier 1 models and escalate to expensive Tier 2 models
only if confidence is low or agreement is poor.

Escalation triggers:
- Average confidence below threshold (default: 6.0)
- High disagreement in rankings (first-place agreement ratio below 50%)
- Low minimum confidence (any model below 4)
"""

from typing import List, Dict, Any, Tuple
from .config_api import (
    get_tier1_models,
    get_tier2_models,
    get_escalation_thresholds,
)


def should_escalate(
    aggregate_confidence: Dict[str, Any],
    stage1_results: List[Dict[str, Any]] = None,
    stage2_results: List[Dict[str, Any]] = None,
    label_to_model: Dict[str, str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine whether to escalate to Tier 2 models based on Tier 1 results.

    Escalation occurs if any of the following conditions are met:
    1. Average confidence is below the confidence threshold
    2. Any model's confidence is below the minimum threshold
    3. First-place agreement ratio is below the agreement threshold

    Args:
        aggregate_confidence: Aggregate confidence statistics from Stage 1
        stage1_results: Optional Stage 1 results for min confidence check
        stage2_results: Optional Stage 2 results for agreement check
        label_to_model: Optional label to model mapping for agreement check

    Returns:
        Tuple of (should_escalate, escalation_info):
        - should_escalate: bool - Whether escalation is recommended
        - escalation_info: dict - Details about the escalation decision
    """
    thresholds = get_escalation_thresholds()
    reasons = []
    metrics = {}

    # Check average confidence
    avg_confidence = aggregate_confidence.get("average", 10)  # Default high if not available
    metrics["average_confidence"] = avg_confidence
    metrics["confidence_threshold"] = thresholds["confidence_threshold"]

    low_avg_confidence = False
    if avg_confidence is not None and avg_confidence < thresholds["confidence_threshold"]:
        low_avg_confidence = True
        reasons.append(f"average confidence ({avg_confidence:.1f}) below threshold ({thresholds['confidence_threshold']})")

    # Check minimum confidence from individual models
    low_min_confidence = False
    if stage1_results:
        confidences = [r.get("confidence") for r in stage1_results if r.get("confidence") is not None]
        if confidences:
            min_confidence = min(confidences)
            metrics["min_confidence"] = min_confidence
            metrics["min_confidence_threshold"] = thresholds["min_confidence_threshold"]

            if min_confidence < thresholds["min_confidence_threshold"]:
                low_min_confidence = True
                reasons.append(f"minimum confidence ({min_confidence}) below threshold ({thresholds['min_confidence_threshold']})")

    # Check first-place agreement ratio (if Stage 2 results available)
    low_agreement = False
    if stage2_results and label_to_model:
        agreement_ratio = _calculate_agreement_ratio(stage2_results, label_to_model)
        metrics["agreement_ratio"] = agreement_ratio
        metrics["agreement_threshold"] = thresholds["agreement_threshold"]

        if agreement_ratio < thresholds["agreement_threshold"]:
            low_agreement = True
            reasons.append(f"agreement ratio ({agreement_ratio:.0%}) below threshold ({thresholds['agreement_threshold']:.0%})")

    should_escalate_result = low_avg_confidence or low_min_confidence or low_agreement

    escalation_info = {
        "should_escalate": should_escalate_result,
        "reasons": reasons if reasons else ["All metrics within acceptable thresholds"],
        "metrics": metrics,
        "triggers": {
            "low_avg_confidence": low_avg_confidence,
            "low_min_confidence": low_min_confidence,
            "low_agreement": low_agreement,
        }
    }

    return should_escalate_result, escalation_info


def _calculate_agreement_ratio(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> float:
    """
    Calculate the first-place agreement ratio from Stage 2 rankings.

    The agreement ratio measures how many models agree on the top-ranked response.
    A high ratio indicates consensus, while a low ratio indicates disagreement.

    Args:
        stage2_results: Results from Stage 2 rankings
        label_to_model: Mapping from labels to model names

    Returns:
        Agreement ratio (0.0 to 1.0)
    """
    if not stage2_results:
        return 1.0  # No data, assume no disagreement

    # Count votes for each response as #1
    first_place_votes = {}
    total_voters = 0

    for ranking in stage2_results:
        parsed = ranking.get("parsed_ranking", [])
        if parsed and len(parsed) > 0:
            first_place = parsed[0]
            first_place_votes[first_place] = first_place_votes.get(first_place, 0) + 1
            total_voters += 1

    if total_voters == 0:
        return 1.0

    # Return the ratio of the most common first-place vote
    max_votes = max(first_place_votes.values()) if first_place_votes else 0
    return max_votes / total_voters


def merge_tier_results(
    tier1_results: List[Dict[str, Any]],
    tier2_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge results from Tier 1 and Tier 2 into a combined result set.

    Marks each result with its tier for display purposes.

    Args:
        tier1_results: Results from Tier 1 models
        tier2_results: Results from Tier 2 models

    Returns:
        Combined list of results with tier information
    """
    merged = []

    for result in tier1_results:
        result_copy = result.copy()
        result_copy["tier"] = 1
        merged.append(result_copy)

    for result in tier2_results:
        result_copy = result.copy()
        result_copy["tier"] = 2
        merged.append(result_copy)

    return merged


def get_tier_info() -> Dict[str, Any]:
    """
    Get information about the tier configuration for display.

    Returns:
        Dict with tier configuration details
    """
    thresholds = get_escalation_thresholds()

    return {
        "tier1_models": get_tier1_models(),
        "tier2_models": get_tier2_models(),
        "thresholds": thresholds,
        "description": {
            "tier1": "Fast, cost-effective models for initial responses",
            "tier2": "Premium, high-capability models for complex queries",
        },
        "escalation_rules": [
            f"Escalate if average confidence < {thresholds['confidence_threshold']}",
            f"Escalate if any model confidence < {thresholds['min_confidence_threshold']}",
            f"Escalate if first-place agreement < {thresholds['agreement_threshold']:.0%}",
        ]
    }


# Tier display information
TIER_INFO = {
    1: {
        "name": "Tier 1",
        "label": "Fast",
        "description": "Cost-effective models for initial assessment",
        "color": "#3b82f6",  # Blue
    },
    2: {
        "name": "Tier 2",
        "label": "Premium",
        "description": "High-capability models for complex queries",
        "color": "#f59e0b",  # Amber
    },
}
