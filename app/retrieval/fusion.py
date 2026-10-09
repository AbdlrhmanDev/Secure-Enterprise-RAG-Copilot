"""Rank fusion for hybrid retrieval."""

from __future__ import annotations

Ranked = list[tuple[str, float]]


def reciprocal_rank_fusion(rankings: dict[str, Ranked], k: int = 60, weights: dict[str, float] | None = None) -> Ranked:
    """RRF: score(d) = sum_r w_r / (k + rank_r(d)). Uses ranks only, so score scales don't matter."""
    weights = weights or {}
    fused: dict[str, float] = {}
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def weighted_score_fusion(rankings: dict[str, Ranked], weights: dict[str, float]) -> Ranked:
    """Min-max normalize each retriever's scores to [0, 1], then take the weighted sum."""
    fused: dict[str, float] = {}
    for name, ranking in rankings.items():
        if not ranking:
            continue
        scores = [score for _, score in ranking]
        low, high = min(scores), max(scores)
        span = high - low
        weight = weights.get(name, 1.0)
        for chunk_id, score in ranking:
            normalized = (score - low) / span if span else 1.0
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * normalized
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
