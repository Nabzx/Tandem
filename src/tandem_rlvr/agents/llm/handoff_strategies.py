from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HandoffStrategy:
    name: str
    description: str
    instruction: str


HANDOFF_STRATEGIES: dict[str, HandoffStrategy] = {
    "minimal_hint": HandoffStrategy(
        name="minimal_hint",
        description="Give a compact operation hint without doing the work.",
        instruction="Give one short hint that helps the junior know what operation to perform. Do not reveal the final answer.",
    ),
    "structured_steps": HandoffStrategy(
        name="structured_steps",
        description="Give a small numbered plan with exact operations but no final answer.",
        instruction="Give 2-3 numbered steps. Include intermediate expressions but avoid the final answer.",
    ),
    "worked_prefix": HandoffStrategy(
        name="worked_prefix",
        description="Carry out the first intermediate step only.",
        instruction="Carry out only the first intermediate step, then say what remains to be done. Do not reveal the final answer.",
    ),
    "verification_hint": HandoffStrategy(
        name="verification_hint",
        description="Explain how the junior can check its result after solving.",
        instruction="Give a concise hint about how the junior can verify the result after solving. Do not reveal the final answer.",
    ),
    "anti_hallucination": HandoffStrategy(
        name="anti_hallucination",
        description="Emphasize task constraints and avoiding extra assumptions.",
        instruction="Restate the task constraints and warn against adding any extra assumptions. Do not reveal the final answer.",
    ),
    "direct_teaching": HandoffStrategy(
        name="direct_teaching",
        description="Teach the relevant general rule without solving the instance.",
        instruction="Explain the relevant general rule needed to solve this task. Do not reveal the final answer.",
    ),
}

DEFAULT_HANDOFF_STRATEGY = "structured_steps"


def get_handoff_strategy(name: str = DEFAULT_HANDOFF_STRATEGY) -> HandoffStrategy:
    try:
        return HANDOFF_STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown handoff strategy: {name}") from exc


def list_handoff_strategy_names() -> list[str]:
    return list(HANDOFF_STRATEGIES)
