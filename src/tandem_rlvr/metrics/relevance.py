from __future__ import annotations

import re
from typing import Any

from tandem_rlvr.tasks.base import Task


SUSPICIOUS_PHRASES = [
    "do not exceed",
    "threshold",
    "remainder",
    "divide by 1",
    "go over 100",
    "external data",
    "not enough information",
]


def compute_relevance_metrics(task: Task, reasoning: str | None) -> dict[str, Any]:
    """Compute transparent relevance and hallucination heuristics."""

    text = (reasoning or "").lower()
    prompt = task.prompt.lower()
    mentions_task_entities = _mentions_task_entities(prompt, text)
    mentions_operation = _mentions_operation(task, text)
    hallucination_flags = [phrase for phrase in SUSPICIOUS_PHRASES if phrase in text]
    irrelevant_constraint_flags = [
        phrase
        for phrase in hallucination_flags
        if phrase in {"do not exceed", "threshold", "go over 100", "external data"}
    ]

    score = 1.0
    if not text.strip():
        score -= 0.6
    if not mentions_task_entities:
        score -= 0.2
    if not mentions_operation:
        score -= 0.25
    if hallucination_flags:
        score -= min(0.4, 0.15 * len(hallucination_flags))

    return {
        "mentions_task_entities": mentions_task_entities,
        "mentions_operation": mentions_operation,
        "irrelevant_constraint_flags": irrelevant_constraint_flags,
        "hallucination_flags": hallucination_flags,
        "relevance_score": max(0.0, min(1.0, score)),
    }


def _mentions_task_entities(prompt: str, reasoning: str) -> bool:
    numbers = re.findall(r"-?\d+", prompt)
    names = re.findall(r"\b[A-Z][a-z]+\b", prompt)
    variables = re.findall(r"\b[a-zA-Z_]\w*\b", prompt)
    entities = set(numbers + [name.lower() for name in names] + [var.lower() for var in variables if var in {"x", "total", "items"}])
    if not entities:
        return bool(reasoning.strip())
    return any(entity in reasoning for entity in entities)


def _mentions_operation(task: Task, reasoning: str) -> bool:
    keywords = _operation_keywords(task.task_type)
    return any(keyword in reasoning for keyword in keywords)


def _operation_keywords(task_type: str) -> list[str]:
    if task_type == "addition":
        return ["+", "add", "sum", "plus"]
    if task_type == "subtraction":
        return ["-", "subtract", "minus", "difference"]
    if task_type == "multiplication":
        return ["*", "multiply", "product", "times"]
    if task_type.startswith("list_"):
        return ["list", "sort", "filter", "even", "odd", "add", "multiply", "reverse", "each"]
    if task_type.startswith("logic_"):
        return ["if", "all", "then", "implies", "transitive", "comparison", "shortest", "taller", "guaranteed"]
    if task_type.startswith("code_trace_"):
        return ["x", "assignment", "loop", "sum", "total", "items", "trace"]
    return []
