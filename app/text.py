"""Small text utilities shared by chunking, lexical retrieval and evaluation."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_WORD_RE = re.compile(r"\w+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it
    its may might must my no not of on or our shall should so than that the their them then there these they
    this those to was we were what when where which who whom why will with would you your""".split()
)


def normalize_text(text: str) -> str:
    """NFKC-normalize, drop control characters and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


def count_tokens(text: str) -> int:
    """Approximate token count (words + punctuation). Model-agnostic and deterministic."""
    return len(_TOKEN_RE.findall(text))


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _stem(token: str) -> str:
    # Deliberately tiny: fold plurals so "refunds" matches "refund".
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Lowercased, stopword-free, lightly stemmed tokens for lexical matching."""
    return [_stem(t) for t in words(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
