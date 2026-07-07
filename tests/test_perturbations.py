import random

from tandem_rlvr.eval.perturbations import (
    drop_numbers,
    inject_irrelevant_sentence,
    light_noise,
    shuffle_sentences,
    truncate_reasoning,
)


def test_truncate_reasoning_removes_tail() -> None:
    assert truncate_reasoning("one two three four", n_words=2) == "one two"
    assert truncate_reasoning("First sentence. Answer sentence.", drop_last_sentence=True) == "First sentence."


def test_drop_numbers_is_seedable() -> None:
    corrupted = drop_numbers("The answer is 123.", rng=random.Random(0), drop_prob=1.0)

    assert corrupted == "The answer is [NUM]."


def test_sentence_perturbations_preserve_text_fragments() -> None:
    reasoning = "First sentence. Second sentence."

    assert "First sentence." in shuffle_sentences(reasoning, rng=random.Random(1))
    assert "irrelevant" in inject_irrelevant_sentence(reasoning, rng=random.Random(1)).lower()
    assert light_noise(reasoning, rng=random.Random(1))
