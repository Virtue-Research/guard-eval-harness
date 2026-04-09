"""Partial-evaluation judgment helpers."""

from __future__ import annotations

_DROP_RATE_THRESHOLD = 0.01


def partial_evaluation_judgment(
    *,
    dropped_count: int,
    total_count: int,
) -> str | None:
    """Classify a partial run as acceptable or flagged.

    Returns ``"acceptable_partial"`` when the drop rate is below 1 %,
    ``"flagged_partial"`` when it is at or above 1 %, and ``None`` when
    there are no dropped samples.
    """
    if dropped_count <= 0 or total_count <= 0:
        return None
    drop_rate = dropped_count / total_count
    if drop_rate < _DROP_RATE_THRESHOLD:
        return "acceptable_partial"
    return "flagged_partial"
