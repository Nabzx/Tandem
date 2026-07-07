from __future__ import annotations

import random
import re


_NUMBER_RE = re.compile(r"-?\d+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def truncate_reasoning(reasoning: str, n_words: int = 6, drop_last_sentence: bool = False) -> str:
    """Remove the final sentence or final N words from a reasoning trace."""

    text = reasoning.strip()
    if text == "":
        return text
    if drop_last_sentence:
        sentences = _split_sentences(text)
        return " ".join(sentences[:-1]).strip() if len(sentences) > 1 else ""

    words = text.split()
    if len(words) <= n_words:
        return ""
    return " ".join(words[:-n_words])


def drop_numbers(reasoning: str, rng: random.Random | None = None, drop_prob: float = 0.7) -> str:
    """Replace sampled numbers with a placeholder token."""

    rng = rng or random.Random()

    def replace(match: re.Match[str]) -> str:
        return "[NUM]" if rng.random() < drop_prob else match.group(0)

    return _NUMBER_RE.sub(replace, reasoning)


def shuffle_sentences(reasoning: str, rng: random.Random | None = None) -> str:
    """Shuffle sentence order while preserving sentence text."""

    rng = rng or random.Random()
    sentences = _split_sentences(reasoning)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def inject_irrelevant_sentence(reasoning: str, rng: random.Random | None = None) -> str:
    """Insert a harmless unrelated sentence into the trace."""

    rng = rng or random.Random()
    distractors = [
        "This note is irrelevant to the task.",
        "This irrelevant note about a notebook does not matter.",
        "Ignore this irrelevant observation.",
    ]
    sentences = _split_sentences(reasoning)
    insert_at = rng.randint(0, len(sentences)) if sentences else 0
    sentences.insert(insert_at, rng.choice(distractors))
    return " ".join(sentences)


def light_noise(reasoning: str, rng: random.Random | None = None) -> str:
    """Apply one simple random perturbation."""

    rng = rng or random.Random()
    choice = rng.choice(["truncate_words", "truncate_sentence", "drop_numbers", "shuffle", "inject"])
    if choice == "truncate_words":
        return truncate_reasoning(reasoning, n_words=rng.randint(3, 8))
    if choice == "truncate_sentence":
        return truncate_reasoning(reasoning, drop_last_sentence=True)
    if choice == "drop_numbers":
        return drop_numbers(reasoning, rng=rng)
    if choice == "shuffle":
        return shuffle_sentences(reasoning, rng=rng)
    return inject_irrelevant_sentence(reasoning, rng=rng)


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]
