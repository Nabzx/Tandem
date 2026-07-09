from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class EpsilonGreedyBandit:
    actions: list[str]
    epsilon: float = 0.1
    seed: int | None = None
    counts: dict[str, int] = field(init=False)
    values: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.counts = {action: 0 for action in self.actions}
        self.values = {action: 0.0 for action in self.actions}

    def select_action(self, context: dict | None = None) -> str:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        return max(self.actions, key=lambda action: (self.values[action], -self.actions.index(action)))

    def update(self, action: str, reward: float, context: dict | None = None) -> None:
        self.counts[action] += 1
        count = self.counts[action]
        self.values[action] += (reward - self.values[action]) / count


@dataclass
class UCB1Bandit:
    actions: list[str]
    exploration: float = 2.0
    counts: dict[str, int] = field(init=False)
    values: dict[str, float] = field(init=False)
    total_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.counts = {action: 0 for action in self.actions}
        self.values = {action: 0.0 for action in self.actions}

    def select_action(self, context: dict | None = None) -> str:
        for action in self.actions:
            if self.counts[action] == 0:
                return action
        total = max(1, self.total_count)
        return max(
            self.actions,
            key=lambda action: self.values[action] + math.sqrt(self.exploration * math.log(total) / self.counts[action]),
        )

    def update(self, action: str, reward: float, context: dict | None = None) -> None:
        self.total_count += 1
        self.counts[action] += 1
        count = self.counts[action]
        self.values[action] += (reward - self.values[action]) / count
