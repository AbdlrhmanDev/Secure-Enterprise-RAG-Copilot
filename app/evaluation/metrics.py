"""Retrieval and answer-quality metrics. Pure functions, no I/O."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

from app.generation.citations import strip_citations
from app.text import content_tokens, split_sentences


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Relevant items in the top k, divided by k (not by the number returned)."""
    return len(set(ranked[:k]) & set(relevant)) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-relevance nDCG."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(set(relevant)), k) + 1))
    return dcg / ideal


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def answer_token_recall(answer: str, expected: str) -> float:
    """Share of the expected answer's content tokens that appear in the answer."""
    expected_tokens = set(content_tokens(expected))
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & set(content_tokens(strip_citations(answer)))) / len(expected_tokens)


def groundedness(answer: str, cited_texts: Sequence[str], threshold: float = 0.7) -> float:
    """Share of answer sentences whose content tokens are (mostly) present in the cited passages.

    A lexical proxy for faithfulness: it catches claims with no support in what was cited, but
    cannot judge paraphrase or logic. Use the LLM judge for that.
    """
    sentences = [s for s in split_sentences(strip_citations(answer)) if content_tokens(s)]
    if not sentences:
        return 0.0
    support = set(content_tokens(" ".join(cited_texts)))
    grounded = 0
    for sentence in sentences:
        tokens = set(content_tokens(sentence))
        if len(tokens & support) / len(tokens) >= threshold:
            grounded += 1
    return grounded / len(sentences)


def citation_precision(cited: Sequence[str], relevant: Collection[str]) -> float:
    """Share of cited chunks that are labeled relevant. An uncited answer scores 0."""
    if not cited:
        return 0.0
    return len([c for c in cited if c in relevant]) / len(cited)
