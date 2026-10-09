"""Answer generators: OpenAI and an offline extractive fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.errors import ServiceUnavailableError, UpstreamError
from app.generation.prompts import (
    ANSWER_SCHEMA,
    INSUFFICIENT_EVIDENCE_ANSWER,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.retrieval.types import ScoredChunk
from app.text import content_tokens, split_sentences

logger = logging.getLogger(__name__)



@dataclass
class Generation:
    answer: str
    insufficient_evidence: bool
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str

    def generate(self, query: str, contexts: list[ScoredChunk]) -> Generation: ...


def openai_request(settings: Settings, *, model: str, system: str, user: str, schema_name: str, schema: dict) -> dict:
    """Chat Completions request with a strict JSON-schema response, shared by generation and the judge."""
    request = {
        "model": model,
        "max_completion_tokens": settings.llm_max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }
    if settings.openai_reasoning_effort:
        request["reasoning_effort"] = settings.openai_reasoning_effort
    return request


class OpenAILLM:
    def __init__(self, settings: Settings):
        import openai

        self._openai = openai
        self._client = openai.OpenAI(api_key=settings.openai_api_key or None)  # None = read the environment
        self._settings = settings
        self.name = settings.openai_model

    def generate(self, query: str, contexts: list[ScoredChunk]) -> Generation:
        openai = self._openai
        request = openai_request(
            self._settings,
            model=self._settings.openai_model,
            system=SYSTEM_PROMPT,
            user=build_user_prompt(query, contexts),
            schema_name="grounded_answer",
            schema=ANSWER_SCHEMA,
        )
        try:
            response = self._client.chat.completions.create(**request)
        except openai.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after", "30")
            raise ServiceUnavailableError(
                "The language model is rate limited, retry shortly", headers={"Retry-After": retry_after}
            ) from exc
        except openai.AuthenticationError as exc:
            raise UpstreamError("Language model credentials were rejected") from exc
        except openai.APIStatusError as exc:
            raise UpstreamError(f"Language model request failed with status {exc.status_code}") from exc
        except openai.APIConnectionError as exc:
            raise UpstreamError("Could not reach the language model") from exc

        usage = response.usage
        tokens = {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }
        choice = response.choices[0]
        if choice.message.refusal:
            logger.warning("llm declined the request", extra={"request_id": response.id})
            return Generation("The language model declined to answer this request.", True, response.model, **tokens)
        try:
            payload = json.loads(choice.message.content or "")
            answer = str(payload["answer"]).strip()
            insufficient = bool(payload["insufficient_evidence"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # Typically finish_reason == "length": raise LLM_MAX_TOKENS (reasoning tokens count against it).
            raise UpstreamError(
                "Language model returned an incomplete answer", details={"finish_reason": choice.finish_reason}
            ) from exc
        return Generation(answer, insufficient, response.model, **tokens)


class ExtractiveLLM:
    """No-API-key fallback: answers by quoting the sentences that best cover the question.

    It cannot paraphrase or combine facts; it exists so the full pipeline (citations, refusal,
    evaluation) runs end to end offline and in CI.
    """

    name = "extractive"
    min_coverage = 0.34

    def generate(self, query: str, contexts: list[ScoredChunk]) -> Generation:
        query_terms = set(content_tokens(query))
        candidates: list[tuple[float, int, str]] = []
        for number, chunk in enumerate(contexts, start=1):
            for sentence in split_sentences(chunk.text):
                terms = set(content_tokens(sentence))
                if terms and query_terms:
                    candidates.append((len(query_terms & terms) / len(query_terms), number, sentence))
        # Best coverage first; ties go to the higher-ranked source.
        candidates.sort(key=lambda c: (-c[0], c[1]))
        if not candidates or candidates[0][0] < self.min_coverage:
            return Generation(INSUFFICIENT_EVIDENCE_ANSWER, True, self.name)
        cutoff = max(self.min_coverage, candidates[0][0] * 0.75)
        picked = [c for c in candidates[:2] if c[0] >= cutoff]
        answer = " ".join(f"{sentence} [{number}]" for _, number, sentence in picked)
        return Generation(answer, False, self.name)


def build_llm(settings: Settings) -> LLM:
    if settings.resolved_llm_provider() == "openai":
        return OpenAILLM(settings)
    return ExtractiveLLM()
