from __future__ import annotations

from tandem_rlvr.rl.bandit import EpsilonGreedyBandit, UCB1Bandit


def create_bandit(name: str, actions: list[str], seed: int | None = None):
    if name == "epsilon_greedy":
        return EpsilonGreedyBandit(actions=actions, epsilon=0.1, seed=seed)
    if name == "ucb1":
        return UCB1Bandit(actions=actions)
    raise ValueError(f"Unsupported bandit: {name}")
