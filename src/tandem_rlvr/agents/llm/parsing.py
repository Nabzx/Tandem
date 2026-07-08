from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import normalize_final_answer


PARSE_STRICT_JSON = "strict_json"
PARSE_EXTRACTED_JSON = "extracted_json"
PARSE_FALLBACK_TEXT = "fallback_text"
PARSE_FAILED = "parse_failed"


@dataclass(frozen=True)
class ParsedLLMOutput:
    reasoning: str
    final_answer: str | None
    metadata: dict[str, Any]


def parse_llm_response(raw_output: str) -> ParsedLLMOutput:
    raw_output = raw_output or ""
    parsed = _try_json(raw_output)
    if isinstance(parsed, dict):
        return _from_json_dict(parsed, raw_output, PARSE_STRICT_JSON)

    extracted = _extract_first_json_object(raw_output)
    if extracted is not None:
        parsed = _try_json(extracted)
        if isinstance(parsed, dict):
            return _from_json_dict(parsed, raw_output, PARSE_EXTRACTED_JSON)
    elif _looks_like_truncated_json(raw_output):
        return ParsedLLMOutput(
            reasoning=raw_output.strip(),
            final_answer="",
            metadata={"raw_output": raw_output, "parse_status": PARSE_FAILED},
        )

    fallback_answer = _fallback_answer(raw_output)
    if fallback_answer is not None:
        return ParsedLLMOutput(
            reasoning=raw_output.strip(),
            final_answer=fallback_answer,
            metadata={"raw_output": raw_output, "parse_status": PARSE_FALLBACK_TEXT},
        )

    return ParsedLLMOutput(
        reasoning=raw_output.strip(),
        final_answer="",
        metadata={"raw_output": raw_output, "parse_status": PARSE_FAILED},
    )


def normalize_llm_answer(task: Task, answer: str | int | float | None) -> str:
    answer_type = task.metadata.get("answer_type", "int")
    if answer is None:
        return ""
    text = str(answer).strip()
    if answer_type == "bool":
        lower = text.lower().strip(" .!")
        if lower in {"yes", "true", "y"}:
            return "true"
        if lower in {"no", "false", "n"}:
            return "false"
    if answer_type == "text":
        return re.sub(r"[^a-z0-9_ -]+", "", text.lower()).strip()
    if answer_type == "list_int":
        return text
    return normalize_final_answer(text)


def _from_json_dict(parsed: dict[str, Any], raw_output: str, parse_status: str) -> ParsedLLMOutput:
    reasoning = parsed.get("reasoning")
    final_answer = parsed.get("final_answer")
    return ParsedLLMOutput(
        reasoning="" if reasoning is None else str(reasoning),
        final_answer=None if final_answer is None else str(final_answer),
        metadata={"raw_output": raw_output, "parse_status": parse_status},
    )


def _try_json(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _fallback_answer(raw_output: str) -> str | None:
    text = raw_output.strip()
    if text == "":
        return None
    answer_match = re.search(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().strip('"')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _looks_like_truncated_json(raw_output: str) -> bool:
    text = raw_output.strip()
    return text.startswith("{") and not text.endswith("}")
