from __future__ import annotations

import json

from tandem_rlvr.tasks.base import Task


JSON_SCHEMA_DIRECT = '{"reasoning": "brief reasoning here", "final_answer": "answer here"}'
JSON_SCHEMA_HANDOFF = '{"reasoning": "partial reasoning here", "final_answer": null}'
FEW_SHOT_EXAMPLES = [
    '{"reasoning": "Compute 7 * 8 = 56.", "final_answer": "56"}',
    '{"reasoning": "Add 4 to each list element.", "final_answer": "[13, 20, 31]"}',
    '{"reasoning": "The implication is transitive.", "final_answer": "true"}',
]


def senior_direct_prompt(task: Task) -> str:
    return _base_prompt(
        role="You are the stronger senior reasoning model.",
        task=task,
        instructions=_direct_instructions(task),
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
            "Use exactly two fields: \"reasoning\" and \"final_answer\".",
            f"Use this schema: {JSON_SCHEMA_HANDOFF}",
        ],
    )


def junior_direct_prompt(task: Task) -> str:
    return _base_prompt(
        role="You are a weaker junior reasoning model.",
        task=task,
        instructions=_direct_instructions(task),
    )


def junior_tandem_prompt(task: Task, senior_reasoning: str) -> str:
    prompt = _base_prompt(
        role="You are a weaker model completing a task using another model's partial reasoning.",
        task=task,
        instructions=[
            "Use the reasoning if helpful, but do not blindly trust it.",
            "Complete the task and provide the final answer.",
            *_direct_instructions(task),
        ],
    )
    return f"{prompt}\n\nSenior partial reasoning:\n{senior_reasoning}"


def _direct_instructions(task: Task) -> list[str]:
    instructions = [
        "Return strict JSON only.",
        "Use exactly two fields: \"reasoning\" and \"final_answer\".",
        "Keep reasoning to one short sentence.",
        "Calculate carefully.",
        "The final_answer must be the shortest valid answer.",
        "Do not include markdown.",
        "Do not include extra text.",
        f"Use this schema: {JSON_SCHEMA_DIRECT}",
    ]
    instructions.extend(_task_specific_instructions(task))
    instructions.append("Short examples:")
    instructions.extend(FEW_SHOT_EXAMPLES)
    return instructions


def _task_specific_instructions(task: Task) -> list[str]:
    if task.task_type in {"addition", "subtraction", "multiplication"}:
        return ["For arithmetic, compute directly and check the operation symbol carefully."]
    if task.task_type.startswith("list_"):
        return ['For list tasks, the final_answer must be a complete Python-style list, e.g. "[1, 2, 3]".']
    if task.metadata.get("answer_type") == "bool" or task.task_type in {"logic_syllogism", "logic_implication"}:
        return ['For true/false logic tasks, final_answer must be exactly "true" or "false".']
    return []


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
