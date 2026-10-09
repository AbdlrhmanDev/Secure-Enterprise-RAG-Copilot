"""Mapping of inline [n] markers in an answer back to the chunks they cite."""

from __future__ import annotations

import re

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str, source_count: int) -> tuple[str, list[int]]:
    """Return the answer with invalid markers removed, plus cited source numbers in order of appearance.

    A marker is only valid if it refers to a source that was actually in the context, so the
    model can never "cite" a chunk the user was not shown.
    """
    cited: list[int] = []

    def replace(match: re.Match) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        valid = [n for n in numbers if 1 <= n <= source_count]
        for number in valid:
            if number not in cited:
                cited.append(number)
        return "".join(f"[{n}]" for n in valid)

    cleaned = _MARKER.sub(replace, answer)
    return re.sub(r"[ \t]+([.,;])", r"\1", cleaned).strip(), cited


def strip_citations(answer: str) -> str:
    return _MARKER.sub("", answer)
