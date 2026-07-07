from __future__ import annotations

import json

from tandem_rlvr.tasks.base import Task


JSON_SCHEMA_DIRECT = '{"reasoning": "brief reasoning here", "final_answer": "answer here"}'
JSON_SCHEMA_HANDOFF = '{"reasoning": "partial reasoning here", "final_answer": null}'


def senior_direct_prompt(task: Task) -> str:
    return _base_prompt(
        role="You are the stronger senior reasoning model.",
        task=task,
        instructions=[
            "Solve the task fully.",
            "Use clear, concise step-by-step reasoning.",
            "Return strict JSON only.",
            f"Use this schema: {JSON_SCHEMA_DIRECT}",
        ],
    )


def senior_handoff_prompt(task: Task) -> str:
    return _base_prompt(
        role="You are the stronger senior reasoning model preparing a handoff for a weaker model.",
        task=task,
        instructions=[
            "Produce useful partial reasoning that makes the task easier to continue.",
            "Do not reveal the final answer.",
            "Avoid private jargon and keep the trace inspectable.",
            "Return strict JSON only.",
            f"Use this schema: {JSON_SCHEMA_HANDOFF}",
        ],
    )


def junior_direct_prompt(task: Task) -> str:
    return _base_prompt(
        role="You are a weaker junior reasoning model.",
        task=task,
        instructions=[
            "Solve the task directly.",
            "Keep reasoning brief.",
            "Return strict JSON only.",
            f"Use this schema: {JSON_SCHEMA_DIRECT}",
        ],
    )


def junior_tandem_prompt(task: Task, senior_reasoning: str) -> str:
    prompt = _base_prompt(
        role="You are a weaker model completing a task using another model's partial reasoning.",
        task=task,
        instructions=[
            "Use the reasoning if helpful, but do not blindly trust it.",
            "Complete the task and provide the final answer.",
            "Return strict JSON only.",
            f"Use this schema: {JSON_SCHEMA_DIRECT}",
        ],
    )
    return f"{prompt}\n\nSenior partial reasoning:\n{senior_reasoning}"


def _base_prompt(role: str, task: Task, instructions: list[str]) -> str:
    metadata = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "difficulty": task.difficulty,
    }
    instruction_text = "\n".join(f"- {instruction}" for instruction in instructions)
    return (
        f"{role}\n\n"
        f"Task metadata:\n{json.dumps(metadata, sort_keys=True)}\n\n"
        f"Task prompt:\n{task.prompt}\n\n"
        f"Instructions:\n{instruction_text}"
    )
