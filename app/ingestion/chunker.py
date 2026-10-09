"""Section-aware, sentence-packing chunker with configurable size and overlap."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers import Block
from app.text import count_tokens, split_sentences


@dataclass
class ChunkDraft:
    text: str
    page: int | None
    section: str | None
    token_count: int


@dataclass
class _Sentence:
    text: str
    page: int | None
    tokens: int


def _sentences(block: Block, chunk_size: int) -> list[_Sentence]:
    out = []
    for sentence in split_sentences(block.text):
        tokens = count_tokens(sentence)
        if tokens <= chunk_size:
            out.append(_Sentence(sentence, block.page, tokens))
            continue
        # A single "sentence" longer than a chunk (tables, code, missing punctuation): hard-split on words.
        parts = sentence.split()
        for start in range(0, len(parts), chunk_size):
            piece = " ".join(parts[start : start + chunk_size])
            out.append(_Sentence(piece, block.page, count_tokens(piece)))
    return out


def chunk_blocks(blocks: list[Block], chunk_size: int = 220, overlap: int = 40) -> list[ChunkDraft]:
    """Pack sentences into chunks of at most `chunk_size` tokens.

    Chunks never cross a section boundary, so every chunk has exactly one section label, and
    consecutive chunks within a section share up to `overlap` tokens of trailing sentences.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[ChunkDraft] = []

    def emit(sentences: list[_Sentence], section: str | None) -> None:
        text = " ".join(s.text for s in sentences)
        chunks.append(ChunkDraft(text, sentences[0].page, section, sum(s.tokens for s in sentences)))

    def pack(sentences: list[_Sentence], section: str | None) -> None:
        current: list[_Sentence] = []
        current_tokens = 0
        fresh = 0  # sentences in `current` that have not been emitted yet
        for sentence in sentences:
            if current and current_tokens + sentence.tokens > chunk_size:
                emit(current, section)
                tail: list[_Sentence] = []
                tail_tokens = 0
                for previous in reversed(current):
                    if tail_tokens + previous.tokens > overlap:
                        break
                    tail.insert(0, previous)
                    tail_tokens += previous.tokens
                # The overlap must leave room for the next sentence.
                while tail and tail_tokens + sentence.tokens > chunk_size:
                    tail_tokens -= tail.pop(0).tokens
                current, current_tokens, fresh = tail, tail_tokens, 0
            current.append(sentence)
            current_tokens += sentence.tokens
            fresh += 1
        if fresh:
            emit(current, section)

    group: list[_Sentence] = []
    group_section: str | None = None
    for block in blocks:
        if group and block.section != group_section:
            pack(group, group_section)
            group = []
        group_section = block.section
        group.extend(_sentences(block, chunk_size))
    if group:
        pack(group, group_section)
    return chunks
