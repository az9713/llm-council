"""Process logging module for Live Process Monitor.

This module provides utilities for generating process events at different
verbosity levels. Events are emitted during the council deliberation process
to show users what's happening behind the scenes.

Verbosity Levels:
- 0: Silent - No process events (just results)
- 1: Basic - Stage transitions only
- 2: Standard - Stage transitions + model-level events
- 3: Verbose - All events including detailed internal operations
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import IntEnum


class Verbosity(IntEnum):
    """Verbosity levels for process logging."""
    SILENT = 0    # No process events
    BASIC = 1     # Stage transitions only
    STANDARD = 2  # Stage + model events
    VERBOSE = 3   # All detailed events


# Event categories for styling in frontend
class EventCategory:
    STAGE = "stage"           # Stage transitions (blue)
    MODEL = "model"           # Model operations (purple)
    INFO = "info"             # General info (gray)
    SUCCESS = "success"       # Success events (green)
    WARNING = "warning"       # Warnings (amber)
    ERROR = "error"           # Errors (red)
    DATA = "data"             # Data/stats (teal)


def create_process_event(
    message: str,
    category: str = EventCategory.INFO,
    level: int = Verbosity.BASIC,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a process event dictionary.

    Args:
        message: Human-readable description of what's happening
        category: Event category for styling (stage, model, info, etc.)
        level: Minimum verbosity level required to show this event
        details: Optional additional data to include

    Returns:
        Process event dictionary ready to be emitted via SSE
    """
    event = {
        "type": "process",
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "category": category,
        "level": level,
    }
    if details:
        event["details"] = details
    return event


def should_emit(event_level: int, current_verbosity: int) -> bool:
    """
    Check if an event should be emitted based on verbosity.

    Args:
        event_level: The minimum verbosity level for this event
        current_verbosity: The user's selected verbosity level

    Returns:
        True if the event should be emitted
    """
    return current_verbosity >= event_level


# ============================================================================
# Pre-defined Event Generators
# ============================================================================


def stage_start(stage_num: int, stage_name: str) -> Dict[str, Any]:
    """Generate a stage start event (Level 1)."""
    return create_process_event(
        message=f"Stage {stage_num}: {stage_name} starting",
        category=EventCategory.STAGE,
        level=Verbosity.BASIC,
        details={"stage": stage_num, "name": stage_name}
    )


def stage_complete(stage_num: int, stage_name: str, duration_ms: Optional[float] = None) -> Dict[str, Any]:
    """Generate a stage complete event (Level 1)."""
    details = {"stage": stage_num, "name": stage_name}
    if duration_ms is not None:
        details["duration_ms"] = round(duration_ms, 0)
    return create_process_event(
        message=f"Stage {stage_num}: {stage_name} complete" + (f" ({duration_ms:.0f}ms)" if duration_ms else ""),
        category=EventCategory.SUCCESS,
        level=Verbosity.BASIC,
        details=details
    )


def model_query_start(model: str, stage: int) -> Dict[str, Any]:
    """Generate a model query start event (Level 2)."""
    model_short = model.split('/')[-1] if '/' in model else model
    return create_process_event(
        message=f"Querying {model_short}...",
        category=EventCategory.MODEL,
        level=Verbosity.STANDARD,
        details={"model": model, "stage": stage}
    )


def model_query_complete(model: str, stage: int, tokens: Optional[int] = None) -> Dict[str, Any]:
    """Generate a model query complete event (Level 2)."""
    model_short = model.split('/')[-1] if '/' in model else model
    message = f"{model_short} responded"
    if tokens:
        message += f" ({tokens} tokens)"
    return create_process_event(
        message=message,
        category=EventCategory.SUCCESS,
        level=Verbosity.STANDARD,
        details={"model": model, "stage": stage, "tokens": tokens}
    )


def model_query_error(model: str, error: str) -> Dict[str, Any]:
    """Generate a model query error event (Level 2)."""
    model_short = model.split('/')[-1] if '/' in model else model
    return create_process_event(
        message=f"{model_short} failed: {error[:50]}",
        category=EventCategory.ERROR,
        level=Verbosity.STANDARD,
        details={"model": model, "error": error}
    )


def models_queried_parallel(models: List[str], stage: int) -> Dict[str, Any]:
    """Generate event for parallel model querying (Level 2)."""
    return create_process_event(
        message=f"Querying {len(models)} models in parallel",
        category=EventCategory.INFO,
        level=Verbosity.STANDARD,
        details={"models": models, "count": len(models), "stage": stage}
    )


def anonymizing_responses(count: int) -> Dict[str, Any]:
    """Generate event for response anonymization (Level 2)."""
    return create_process_event(
        message=f"Anonymizing {count} responses for peer review",
        category=EventCategory.INFO,
        level=Verbosity.STANDARD,
        details={"response_count": count}
    )


def parsing_rankings(model: str) -> Dict[str, Any]:
    """Generate event for parsing rankings (Level 3)."""
    model_short = model.split('/')[-1] if '/' in model else model
    return create_process_event(
        message=f"Parsing ranking from {model_short}",
        category=EventCategory.DATA,
        level=Verbosity.VERBOSE,
        details={"model": model}
    )


def ranking_parsed(model: str, ranking: List[str]) -> Dict[str, Any]:
    """Generate event for parsed ranking result (Level 3)."""
    model_short = model.split('/')[-1] if '/' in model else model
    return create_process_event(
        message=f"{model_short} ranked: {' > '.join(ranking) if ranking else 'failed to parse'}",
        category=EventCategory.DATA,
        level=Verbosity.VERBOSE,
        details={"model": model, "ranking": ranking}
    )


def aggregate_rankings_calculated(top_model: str, avg_rank: float) -> Dict[str, Any]:
    """Generate event for aggregate rankings calculation (Level 2)."""
    model_short = top_model.split('/')[-1] if '/' in top_model else top_model
    return create_process_event(
        message=f"Top ranked: {model_short} (avg rank: {avg_rank:.2f})",
        category=EventCategory.DATA,
        level=Verbosity.STANDARD,
        details={"top_model": top_model, "average_rank": avg_rank}
    )


def confidence_parsed(model: str, confidence: Optional[int]) -> Dict[str, Any]:
    """Generate event for confidence parsing (Level 3)."""
    model_short = model.split('/')[-1] if '/' in model else model
    if confidence is not None:
        return create_process_event(
            message=f"{model_short} confidence: {confidence}/10",
            category=EventCategory.DATA,
            level=Verbosity.VERBOSE,
            details={"model": model, "confidence": confidence}
        )
    else:
        return create_process_event(
            message=f"{model_short}: no confidence score found",
            category=EventCategory.WARNING,
            level=Verbosity.VERBOSE,
            details={"model": model, "confidence": None}
        )


def aggregate_confidence_calculated(average: float, count: int, total: int) -> Dict[str, Any]:
    """Generate event for aggregate confidence calculation (Level 2)."""
    return create_process_event(
        message=f"Aggregate confidence: {average:.1f}/10 ({count}/{total} models reported)",
        category=EventCategory.DATA,
        level=Verbosity.STANDARD,
        details={"average": average, "count": count, "total": total}
    )


def chairman_synthesizing(chairman_model: str) -> Dict[str, Any]:
    """Generate event for chairman synthesis start (Level 2)."""
    model_short = chairman_model.split('/')[-1] if '/' in chairman_model else chairman_model
    return create_process_event(
        message=f"Chairman {model_short} synthesizing final answer",
        category=EventCategory.MODEL,
        level=Verbosity.STANDARD,
        details={"model": chairman_model}
    )


def prompt_prepared(stage: int, char_count: int) -> Dict[str, Any]:
    """Generate event for prompt preparation (Level 3)."""
    return create_process_event(
        message=f"Stage {stage} prompt prepared ({char_count:,} chars)",
        category=EventCategory.INFO,
        level=Verbosity.VERBOSE,
        details={"stage": stage, "char_count": char_count}
    )


def cost_calculated(stage: str, cost: float) -> Dict[str, Any]:
    """Generate event for cost calculation (Level 3)."""
    return create_process_event(
        message=f"{stage} cost: ${cost:.4f}",
        category=EventCategory.DATA,
        level=Verbosity.VERBOSE,
        details={"stage": stage, "cost": cost}
    )


def total_cost_calculated(total: float) -> Dict[str, Any]:
    """Generate event for total cost (Level 2)."""
    return create_process_event(
        message=f"Total cost: ${total:.4f}",
        category=EventCategory.DATA,
        level=Verbosity.STANDARD,
        details={"total_cost": total}
    )


def analytics_recorded() -> Dict[str, Any]:
    """Generate event for analytics recording (Level 3)."""
    return create_process_event(
        message="Analytics recorded for performance tracking",
        category=EventCategory.INFO,
        level=Verbosity.VERBOSE,
    )


def title_generated(title: str) -> Dict[str, Any]:
    """Generate event for title generation (Level 2)."""
    return create_process_event(
        message=f"Title generated: \"{title}\"",
        category=EventCategory.SUCCESS,
        level=Verbosity.STANDARD,
        details={"title": title}
    )


def streaming_started(model: str) -> Dict[str, Any]:
    """Generate event for streaming start (Level 3)."""
    model_short = model.split('/')[-1] if '/' in model else model
    return create_process_event(
        message=f"{model_short} streaming started",
        category=EventCategory.INFO,
        level=Verbosity.VERBOSE,
        details={"model": model}
    )


def token_received(model: str, token_count: int) -> Dict[str, Any]:
    """Generate periodic event for tokens received (Level 3)."""
    model_short = model.split('/')[-1] if '/' in model else model
    return create_process_event(
        message=f"{model_short}: {token_count} tokens received",
        category=EventCategory.DATA,
        level=Verbosity.VERBOSE,
        details={"model": model, "token_count": token_count}
    )
