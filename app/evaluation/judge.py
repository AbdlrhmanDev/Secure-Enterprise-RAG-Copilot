"""Optional LLM judge for answer correctness and faithfulness (requires OPENAI_API_KEY)."""

from __future__ import annotations

import json

from app.config import Settings
from app.generation.llm import openai_request

JUDGE_SYSTEM = """\
You grade answers produced by a retrieval-augmented question answering system. You are given the \
question, a reference answer written by the dataset author, the system's answer, and the passages the \
system cited. Grade two things independently:

- correct: the system's answer conveys the same facts as the reference answer. Extra detail is fine as \
long as it does not contradict the reference. An answer that declines or hedges without stating the \
fact is not correct.
- faithful: every factual claim in the system's answer is supported by the cited passages. Judge only \
against the passages, not against what you know or against the reference.

Return JSON with boolean fields "correct" and "faithful"."""

_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "faithful": {"type": "boolean"}},
    "required": ["correct", "faithful"],
    "additionalProperties": False,
}


class LLMJudge:
    def __init__(self, settings: Settings):
        import openai

        self._client = openai.OpenAI(api_key=settings.openai_api_key or None)
        self._settings = settings
        self.model = settings.judge_model or settings.openai_model

    def grade(self, *, question: str, expected: str, answer: str, cited_texts: list[str]) -> dict[str, bool]:
        passages = "\n".join(f"<passage>\n{text}\n</passage>" for text in cited_texts) or "(no passages cited)"
        prompt = (
            f"<question>\n{question}\n</question>\n<reference_answer>\n{expected}\n</reference_answer>\n"
            f"<system_answer>\n{answer}\n</system_answer>\n<cited_passages>\n{passages}\n</cited_passages>"
        )
        request = openai_request(
            self._settings, model=self.model, system=JUDGE_SYSTEM, user=prompt, schema_name="grade", schema=_SCHEMA
        )
        message = self._client.chat.completions.create(**request).choices[0].message
        try:
            payload = json.loads(message.content or "{}")
        except json.JSONDecodeError:
            payload = {}
        return {"correct": bool(payload.get("correct")), "faithful": bool(payload.get("faithful"))}
