"""Removal of repeated page furniture (running headers/footers, page numbers)."""

from __future__ import annotations

import re
from collections import Counter

_EDGE_LINES = 2  # how many lines at the top/bottom of each page are header/footer candidates
_MIN_PAGES = 3
_REPEAT_RATIO = 0.6


def _normalize(line: str) -> str:
    # "Page 3 of 12" and "Page 4 of 12" must collapse to the same key.
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    """Drop lines that recur at the top or bottom of most pages.

    `pages` is a list of pages, each a list of non-empty text lines. Documents shorter than
    three pages are returned unchanged: there is not enough evidence to call a line furniture.
    """
    if len(pages) < _MIN_PAGES:
        return pages

    counts: Counter[str] = Counter()
    for lines in pages:
        edges = {_normalize(line) for line in lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]}
        counts.update(edges)

    threshold = max(_MIN_PAGES, int(len(pages) * _REPEAT_RATIO))
    furniture = {key for key, n in counts.items() if n >= threshold and key}
    if not furniture:
        return pages

    cleaned = []
    for lines in pages:
        kept = []
        for i, line in enumerate(lines):
            at_edge = i < _EDGE_LINES or i >= len(lines) - _EDGE_LINES
            if at_edge and _normalize(line) in furniture:
                continue
            kept.append(line)
        cleaned.append(kept)
    return cleaned
