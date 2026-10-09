"""The OpenAI generator against a stubbed client: request shape, parsing, refusal and truncation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.errors import UpstreamError
from app.generation.llm import OpenAILLM
from app.retrieval.types import ScoredChunk

CONTEXT = [ScoredChunk("c1", "d1", "refund-policy.md", "Refund Policy", "Refunds above 2,000 USD need a director.", 1, "Approvals", 0.9)]


def _llm(monkeypatch, message, finish_reason="stop"):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAILLM(get_settings())
    calls = []

    def create(**request):
        calls.append(request)
        return SimpleNamespace(
            id="chatcmpl-test",
            model="gpt-test",
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        )

    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm, calls


def test_answer_is_parsed_and_request_uses_strict_json_schema(monkeypatch):
    content = json.dumps({"answer": "A director must approve it [1].", "insufficient_evidence": False})
    llm, calls = _llm(monkeypatch, SimpleNamespace(content=content, refusal=None))
    result = llm.generate("Who approves large refunds?", CONTEXT)

    assert result.answer == "A director must approve it [1]." and not result.insufficient_evidence
    assert (result.input_tokens, result.output_tokens, result.model) == (120, 30, "gpt-test")
    request = calls[0]
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["messages"][0]["role"] == "system"
    assert "Refunds above 2,000 USD" in request["messages"][1]["content"]
    assert "max_completion_tokens" in request and "reasoning_effort" not in request


def test_model_refusal_becomes_an_insufficient_evidence_answer(monkeypatch):
    llm, _ = _llm(monkeypatch, SimpleNamespace(content=None, refusal="I can't help with that."))
    assert llm.generate("q", CONTEXT).insufficient_evidence is True


def test_truncated_output_raises_a_structured_upstream_error(monkeypatch):
    llm, _ = _llm(monkeypatch, SimpleNamespace(content='{"answer": "A dir', refusal=None), finish_reason="length")
    with pytest.raises(UpstreamError) as error:
        llm.generate("q", CONTEXT)
    assert error.value.details == {"finish_reason": "length"}
