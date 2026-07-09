from __future__ import annotations

import random
from collections.abc import Sequence

from tandem_rlvr.tasks.base import Task


SUPPORTED_SPLITS = ("train", "id_eval", "ood_eval", "stress_eval")
SUPPORTED_TASK_FAMILIES = ("arithmetic", "list", "logic", "code")


def build_split_benchmark(
    num_tasks_per_split: int,
    splits: Sequence[str],
    seed: int = 0,
    task_families: Sequence[str] = SUPPORTED_TASK_FAMILIES,
) -> list[Task]:
    unknown_splits = set(splits) - set(SUPPORTED_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unsupported splits: {sorted(unknown_splits)}")
    unknown_families = set(task_families) - set(SUPPORTED_TASK_FAMILIES)
    if unknown_families:
        raise ValueError(f"Unsupported task families: {sorted(unknown_families)}")
    if num_tasks_per_split < 0:
        raise ValueError("num_tasks_per_split must be non-negative")

    rng = random.Random(seed)
    tasks: list[Task] = []
    for split in splits:
        for index in range(num_tasks_per_split):
            family = task_families[index % len(task_families)]
            task_rng = random.Random(rng.randint(0, 10_000_000))
            task = _generate_family_task(family, split, index + 1, task_rng)
            tasks.append(task)
    return tasks


def _generate_family_task(family: str, split: str, index: int, rng: random.Random) -> Task:
    if family == "arithmetic":
        return _arithmetic_task(split, index, rng)
    if family == "list":
        return _list_task(split, index, rng)
    if family == "logic":
        return _logic_task(split, index, rng)
    if family == "code":
        return _code_task(split, index, rng)
    raise ValueError(f"Unsupported task family: {family}")


def _base_metadata(split: str, family: str, ood_type: str) -> dict[str, str]:
    return {
        "split": split,
        "distribution": "id" if split in {"train", "id_eval"} else "ood" if split == "ood_eval" else "stress",
        "ood_type": ood_type,
        "task_family": family,
    }


def _difficulty(split: str) -> str:
    return "easy" if split in {"train", "id_eval"} else "medium" if split == "ood_eval" else "hard"


def _arithmetic_task(split: str, index: int, rng: random.Random) -> Task:
    if split in {"train", "id_eval"}:
        op = rng.choice(["+", "-", "*"])
        left = rng.randint(0, 12)
        right = rng.randint(0, 12)
        if op == "-" and right > left:
            left, right = right, left
        prompt = f"What is {left} {op} {right}?"
        answer = _compute_arithmetic(left, right, op)
        task_type = {"+": "addition", "-": "subtraction", "*": "multiplication"}[op]
        metadata = {"left": left, "right": right, "operator": op, **_base_metadata(split, "arithmetic", "id_small_positive")}
    elif split == "ood_eval":
        variant = rng.choice(["larger_numbers", "negative_result", "zero_one_multiply", "two_step"])
        if variant == "two_step":
            a, b, c = rng.randint(10, 60), rng.randint(10, 60), rng.randint(2, 9)
            prompt = f"What is ({a} + {b}) * {c}?"
            answer = (a + b) * c
            task_type = "arithmetic_two_step"
            metadata = {"left": a, "right": b, "third": c, "operator": "+,*", **_base_metadata(split, "arithmetic", variant)}
        else:
            op = rng.choice(["+", "-", "*"])
            left = rng.randint(20, 200)
            right = rng.randint(20, 200)
            if variant == "negative_result":
                left, right, op = rng.randint(1, 40), rng.randint(50, 120), "-"
            if variant == "zero_one_multiply":
                left, right, op = rng.randint(0, 1), rng.randint(20, 200), "*"
            prompt = f"What is   {left}   {op}   {right}?"
            answer = _compute_arithmetic(left, right, op)
            task_type = {"+": "addition", "-": "subtraction", "*": "multiplication"}[op]
            metadata = {"left": left, "right": right, "operator": op, **_base_metadata(split, "arithmetic", variant)}
    else:
        variant = rng.choice(["stress_two_step", "distractor_wording", "edge_case"])
        if variant == "stress_two_step":
            a, b, c = rng.randint(100, 500), rng.randint(20, 200), rng.randint(2, 12)
            prompt = f"Ignore the label A. Compute ({a} - {b}) + {c}."
            answer = (a - b) + c
            task_type = "arithmetic_two_step"
            metadata = {"left": a, "right": b, "third": c, "operator": "-,+", **_base_metadata(split, "arithmetic", variant)}
        elif variant == "edge_case":
            n = rng.randint(50, 500)
            op = rng.choice(["*", "+", "-"])
            if op == "*":
                left, right = 0, n
            elif op == "+":
                left, right = n, 0
            else:
                left, right = rng.randint(1, 50), n
            prompt = f"What is {left} {op} {right}?"
            answer = _compute_arithmetic(left, right, op)
            task_type = {"+": "addition", "-": "subtraction", "*": "multiplication"}[op]
            metadata = {"left": left, "right": right, "operator": op, **_base_metadata(split, "arithmetic", variant)}
        else:
            left, right = rng.randint(100, 900), rng.randint(100, 900)
            prompt = f"The account number is irrelevant. What is {left} - {right}?"
            answer = left - right
            task_type = "subtraction"
            metadata = {"left": left, "right": right, "operator": "-", **_base_metadata(split, "arithmetic", variant)}

    return Task(_task_id(split, "arith", index), task_type, prompt, str(answer), _difficulty(split), metadata)


def _list_task(split: str, index: int, rng: random.Random) -> Task:
    if split in {"train", "id_eval"}:
        values = [rng.randint(0, 12) for _ in range(rng.randint(3, 5))]
        variant = rng.choice(["sort", "filter_even", "map_add", "reverse"])
    elif split == "ood_eval":
        variant = rng.choice(["long_negative", "duplicates", "empty_filter", "already_sorted"])
        values = [rng.randint(-20, 20) for _ in range(rng.randint(6, 9))]
        if variant == "duplicates":
            repeated = rng.randint(-5, 5)
            values[:3] = [repeated, repeated, repeated]
        if variant == "empty_filter":
            values = [2 * rng.randint(0, 10) + 1 for _ in range(6)]
        if variant == "already_sorted":
            values = sorted(values)
    else:
        variant = rng.choice(["mixed_repeated", "empty_list", "reverse_sorted"])
        values = [rng.choice([-10, -5, 0, 5, 10]) for _ in range(rng.randint(8, 12))]
        if variant == "empty_list":
            values = []
        if variant == "reverse_sorted":
            values = sorted(values, reverse=True)

    operation = rng.choice(["sort", "filter_even", "filter_odd", "map_multiply", "reverse"])
    if variant in {"empty_filter"}:
        operation = "filter_even"
    answer, prompt, task_type, operation_meta = _list_operation(values, operation, rng)
    metadata = {
        "values": values,
        "operation": operation_meta,
        "answer_type": "list_int",
        "answer_value": answer,
        **_base_metadata(split, "list", variant),
    }
    return Task(_task_id(split, "list", index), task_type, prompt, str(answer), _difficulty(split), metadata)


def _logic_task(split: str, index: int, rng: random.Random) -> Task:
    if split in {"train", "id_eval"}:
        variant = rng.choice(["simple_syllogism", "direct_comparison"])
    elif split == "ood_eval":
        variant = rng.choice(["false_syllogism", "reversed_comparison", "negated_implication"])
    else:
        variant = rng.choice(["distractor_entities", "longer_chain"])

    if variant in {"simple_syllogism", "false_syllogism", "distractor_entities"}:
        a, b, c = rng.sample(["blickets", "daxes", "wugs", "lorps", "mips"], 3)
        if variant == "false_syllogism":
            prompt = f"If all {a} are {b}, and no {c} are {b}, are all {a} {c}?"
            answer = "false"
        else:
            distractor = " Some zeps are unrelated." if variant == "distractor_entities" else ""
            prompt = f"If all {a} are {b}, and all {b} are {c},{distractor} are all {a} {c}?"
            answer = "true"
        task_type = "logic_syllogism"
    elif variant in {"direct_comparison", "reversed_comparison", "longer_chain"}:
        names = ["Alice", "Bob", "Charlie", "Dana"] if variant == "longer_chain" else ["Alice", "Bob", "Charlie"]
        prompt = " ".join(f"{names[i]} is taller than {names[i + 1]}." for i in range(len(names) - 1))
        if variant == "reversed_comparison":
            prompt += " Who is tallest?"
            answer = names[0]
        else:
            prompt += " Who is shortest?"
            answer = names[-1]
        task_type = "logic_comparison"
    else:
        prompt = "If the switch is on, then the lamp is lit. The lamp is not lit. Is the switch on?"
        answer = "false"
        task_type = "logic_implication"

    metadata = {
        "answer_type": "bool" if answer in {"true", "false"} else "text",
        "answer_value": answer == "true" if answer in {"true", "false"} else answer,
        **_base_metadata(split, "logic", variant),
    }
    return Task(_task_id(split, "logic", index), task_type, prompt, answer, _difficulty(split), metadata)


def _code_task(split: str, index: int, rng: random.Random) -> Task:
    if split in {"train", "id_eval"}:
        variant = rng.choice(["assignment_add", "loop_sum"])
    elif split == "ood_eval":
        variant = rng.choice(["multiplication_update", "negative_values", "empty_loop", "overwrite"])
    else:
        variant = rng.choice(["multiple_updates", "longer_loop", "off_by_one_style"])

    if variant == "assignment_add":
        start, addend = rng.randint(1, 8), rng.randint(1, 8)
        answer = start + addend
        code = f"x = {start}\nx = x + {addend}"
        question = "What is x?"
        task_type = "code_trace_assignment"
    elif variant == "multiplication_update":
        start, factor = rng.randint(2, 9), rng.randint(2, 6)
        answer = start * factor
        code = f"x = {start}\nx = x * {factor}"
        question = "What is x?"
        task_type = "code_trace_assignment"
    elif variant == "overwrite":
        first, second, addend = rng.randint(1, 9), rng.randint(10, 20), rng.randint(1, 5)
        answer = second + addend
        code = f"x = {first}\nx = {second}\nx = x + {addend}"
        question = "What is x?"
        task_type = "code_trace_assignment"
    elif variant in {"loop_sum", "negative_values", "empty_loop", "longer_loop"}:
        length = 0 if variant == "empty_loop" else 3 if variant == "loop_sum" else 5 if variant == "negative_values" else 8
        low = -8 if variant == "negative_values" else 1
        high = 12
        items = [rng.randint(low, high) for _ in range(length)]
        answer = sum(items)
        code = f"items = {items}\ntotal = 0\nfor x in items:\n    total += x"
        question = "What is total?"
        task_type = "code_trace_loop_sum"
    else:
        start, addend, factor = rng.randint(1, 10), rng.randint(1, 10), rng.randint(2, 5)
        if variant == "off_by_one_style":
            answer = (start + addend) * factor
            code = f"x = {start}\nx = x + {addend}\nx = x * {factor}\n# range end is irrelevant here"
        else:
            answer = ((start + addend) * factor) - 1
            code = f"x = {start}\nx = x + {addend}\nx = x * {factor}\nx = x - 1"
        question = "What is x?"
        task_type = "code_trace_assignment"

    prompt = f"Trace this code:\n```python\n{code}\n```\n{question}"
    metadata = {"answer_type": "int", "answer_value": answer, **_base_metadata(split, "code", variant)}
    return Task(_task_id(split, "code", index), task_type, prompt, str(answer), _difficulty(split), metadata)


def _compute_arithmetic(left: int, right: int, op: str) -> int:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    raise ValueError(f"Unsupported operation: {op}")


def _list_operation(values: list[int], operation: str, rng: random.Random) -> tuple[list[int], str, str, str]:
    if operation == "sort":
        return sorted(values), f"Given the list {values}, sort it in ascending order.", "list_sort", "sort"
    if operation == "filter_even":
        return [value for value in values if value % 2 == 0], f"Given the list {values}, return only the even numbers.", "list_filter_even", "filter_even"
    if operation == "filter_odd":
        return [value for value in values if value % 2 != 0], f"Given the list {values}, return only the odd numbers.", "list_filter_odd", "filter_odd"
    if operation == "map_multiply":
        factor = rng.randint(2, 4)
        return [value * factor for value in values], f"Given the list {values}, multiply each element by {factor}.", "list_map_multiply", f"map_multiply:{factor}"
    return list(reversed(values)), f"Given the list {values}, reverse the list.", "list_reverse", "reverse"


def _task_id(split: str, prefix: str, index: int) -> str:
    return f"{split}-{prefix}-{index:06d}"
