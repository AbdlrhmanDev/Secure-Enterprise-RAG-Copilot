"""Unit tests for the pure building blocks: chunking, parsing, cleaning, fusion, metrics, citations."""

from __future__ import annotations

import pytest

from app.evaluation import metrics as m
from app.generation.citations import extract_citations
from app.ingestion.chunker import chunk_blocks
from app.ingestion.cleaner import strip_repeated_lines
from app.ingestion.parsers import Block, ParseError, parse_document
from app.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.text import count_tokens
from scripts.generate_corpus import render_docx, render_html, render_markdown, render_pdf, render_txt

SECTIONS = [
    ("Approval Thresholds", "Refunds above 2,000 USD need director approval. Smaller refunds need a team lead."),
    ("Refund Window", "Customers may request a refund within 45 days of the invoice date."),
]


# --- chunker ---------------------------------------------------------------------------------


def test_chunks_respect_size_and_keep_all_text():
    text = " ".join(f"Sentence number {i} has exactly seven tokens." for i in range(60))
    chunks = chunk_blocks([Block(text, 1, "S")], chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(f"Sentence number {i} has" in joined for i in range(60))


def test_chunks_overlap_within_a_section():
    text = " ".join(f"Fact {i} is stated here." for i in range(40))
    chunks = chunk_blocks([Block(text, None, "S")], chunk_size=40, overlap=12)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = current.text.split(". ")[0]
        assert first_sentence in previous.text


def test_chunks_never_cross_sections_and_carry_metadata():
    blocks = [Block("Alpha one. Alpha two.", 1, "A"), Block("Beta one. Beta two.", 2, "B")]
    chunks = chunk_blocks(blocks, chunk_size=100, overlap=10)
    assert [(c.section, c.page) for c in chunks] == [("A", 1), ("B", 2)]
    assert "Beta" not in chunks[0].text


def test_oversized_sentence_is_split():
    chunks = chunk_blocks([Block("word " * 500, None, None)], chunk_size=100, overlap=0)
    assert len(chunks) == 5 and all(c.token_count <= 100 for c in chunks)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_blocks([Block("x.", None, None)], chunk_size=10, overlap=10)


# --- parsers ---------------------------------------------------------------------------------


def _render(suffix: str, tmp_path) -> bytes:
    if suffix in (".pdf", ".docx"):
        path = tmp_path / f"doc{suffix}"
        (render_pdf if suffix == ".pdf" else render_docx)("Refund Policy", SECTIONS, path)
        return path.read_bytes()
    return {".md": render_markdown, ".txt": render_txt, ".html": render_html}[suffix]("Refund Policy", SECTIONS)


@pytest.mark.parametrize("suffix", [".pdf", ".docx", ".md", ".html", ".txt"])
def test_every_format_yields_text_sections_and_title(suffix, tmp_path):
    parsed = parse_document(f"refund-policy{suffix}", _render(suffix, tmp_path))
    by_section = {b.section: b.text for b in parsed.blocks}
    assert "45 days of the invoice date" in by_section["Refund Window"]
    assert "director approval" in by_section["Approval Thresholds"]
    assert parsed.title == "Refund Policy"
    assert parsed.page_count >= 1
    if suffix == ".pdf":
        assert all(b.page == 1 for b in parsed.blocks)
    if suffix == ".html":
        assert not any("Intranet home" in b.text or "Do not distribute" in b.text for b in parsed.blocks)


def test_pdf_running_header_and_footer_are_removed(tmp_path):
    long_sections = [(f"Topic {i}", "This paragraph fills the page with ordinary text. " * 60) for i in range(8)]
    path = tmp_path / "long.pdf"
    render_pdf("Long Report", long_sections, path)
    parsed = parse_document("long.pdf", path.read_bytes())
    assert parsed.page_count >= 3
    text = " ".join(b.text for b in parsed.blocks)
    assert "Internal Document" not in text
    assert "Page 2" not in text


def test_unsupported_and_empty_files_raise_parse_error():
    with pytest.raises(ParseError):
        parse_document("malware.exe", b"MZ")
    with pytest.raises(ParseError):
        parse_document("empty.txt", b"   \n  ")
    with pytest.raises(ParseError):
        parse_document("broken.pdf", b"this is not a pdf")


def test_strip_repeated_lines_keeps_short_documents_untouched():
    pages = [["Header", "body a"], ["Header", "body b"]]
    assert strip_repeated_lines(pages) == pages
    bodies = [[f"{word} opening.", f"{word} middle.", f"{word} closing."] for word in ("Alpha", "Bravo", "Charlie", "Delta")]
    pages = [["ACME Confidential", *body, f"Page {i} of 4"] for i, body in enumerate(bodies, start=1)]
    assert strip_repeated_lines(pages) == bodies


# --- fusion ----------------------------------------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers():
    rankings = {"bm25": [("a", 9.0), ("b", 5.0), ("c", 1.0)], "dense": [("b", 0.9), ("d", 0.8), ("a", 0.1)]}
    fused = reciprocal_rank_fusion(rankings, k=60)
    assert [chunk_id for chunk_id, _ in fused][:2] == ["b", "a"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_weights_shift_the_ranking():
    rankings = {"bm25": [("a", 1.0)], "dense": [("b", 1.0)]}
    assert reciprocal_rank_fusion(rankings, weights={"bm25": 2.0, "dense": 1.0})[0][0] == "a"
    assert reciprocal_rank_fusion(rankings, weights={"bm25": 1.0, "dense": 2.0})[0][0] == "b"


def test_weighted_fusion_normalizes_score_scales():
    rankings = {"bm25": [("a", 30.0), ("b", 10.0)], "dense": [("b", 0.9), ("a", 0.5)]}
    fused = dict(weighted_score_fusion(rankings, {"bm25": 0.5, "dense": 0.5}))
    assert fused == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}


# --- metrics ---------------------------------------------------------------------------------


def test_retrieval_metrics():
    ranked, relevant = ["x", "a", "y", "b"], {"a", "b", "c"}
    assert m.recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)
    assert m.recall_at_k(ranked, relevant, 4) == pytest.approx(2 / 3)
    assert m.precision_at_k(ranked, relevant, 4) == pytest.approx(0.5)
    assert m.reciprocal_rank(ranked, relevant) == pytest.approx(0.5)
    assert m.reciprocal_rank(["x"], relevant) == 0.0
    assert m.ndcg_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.0)
    assert 0 < m.ndcg_at_k(ranked, relevant, 4) < 1


def test_percentile_interpolates():
    assert m.percentile([10, 20, 30, 40], 50) == pytest.approx(25)
    assert m.percentile([10, 20, 30, 40], 100) == 40
    assert m.percentile([], 95) == 0.0


def test_answer_quality_metrics():
    passage = "Refunds above 2,000 USD must be approved by the Director of Customer Operations."
    assert m.answer_token_recall("It is approved by the Director of Customer Operations [1].", "Director of Customer Operations") == 1.0
    assert m.groundedness("Refunds above 2,000 USD must be approved by the Director [1].", [passage]) == 1.0
    assert m.groundedness("The moon is made of cheese [1].", [passage]) == 0.0
    assert m.citation_precision(["c1", "c2"], {"c1"}) == 0.5
    assert m.citation_precision([], {"c1"}) == 0.0


# --- citations -------------------------------------------------------------------------------


def test_citations_are_extracted_in_order_and_invalid_markers_dropped():
    answer, cited = extract_citations("Limit is 200 USD [2]. Director approves [1, 2] and [7].", source_count=3)
    assert cited == [2, 1]
    assert "[7]" not in answer and "[1][2]" in answer


def test_token_count_is_deterministic():
    assert count_tokens("Refunds above 2,000 USD.") == 7
