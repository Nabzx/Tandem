from __future__ import annotations

import pytest

from tandem_rlvr.agents.llm import (
    OllamaBackendUnavailable,
    OllamaGenerationConfig,
    OllamaGenerationTimeout,
    OllamaJuniorAgent,
    OllamaModelNotFound,
    OllamaSeniorAgent,
)


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def list(self) -> dict[str, list[object]]:
        return {"models": []}

    def chat(self, model: str, messages: list[dict[str, str]], options: dict[str, object], format: str) -> dict[str, object]:
        self.prompts.append(messages[-1]["content"])
        return {"message": {"content": self.content}}


class FailingOllamaClient:
    def list(self) -> None:
        raise RuntimeError("not running")


class TimeoutOllamaClient:
    def list(self) -> dict[str, list[object]]:
        return {"models": [{"name": "fake-model"}]}

    def chat(self, model: str, messages: list[dict[str, str]], options: dict[str, object], format: str) -> dict[str, object]:
        raise RuntimeError("timed out")


class ModelListClient:
    def __init__(self, models: list[str]) -> None:
        self.models = models

    def list(self) -> dict[str, list[object]]:
        return {"models": [{"name": model} for model in self.models]}


def test_ollama_senior_response_object_from_mocked_client() -> None:
    client = FakeOllamaClient('{"reasoning": "Add 2 and 3.", "final_answer": "5"}')
    agent = OllamaSeniorAgent(OllamaGenerationConfig("fake-model"), client=client)
    task = _task()

    response = agent.answer(task)

    assert response.reasoning == "Add 2 and 3."
    assert response.final_answer == "5"
    assert response.metadata["parse_status"] == "strict_json"
    assert response.metadata["model_name"] == "fake-model"


def test_ollama_junior_uses_tandem_prompt_with_context() -> None:
    client = FakeOllamaClient('{"reasoning": "Continue.", "final_answer": "5"}')
    agent = OllamaJuniorAgent(OllamaGenerationConfig("fake-model"), client=client)

    agent.answer(_task(), context="Senior says add the operands.")

    assert "Senior says add the operands." in client.prompts[-1]


def test_ollama_unavailable_has_clear_error() -> None:
    with pytest.raises(OllamaBackendUnavailable, match="Ollama backend unavailable"):
        OllamaSeniorAgent(OllamaGenerationConfig("fake-model"), client=FailingOllamaClient())


def test_ollama_timeout_has_timeout_message() -> None:
    agent = OllamaSeniorAgent(OllamaGenerationConfig("fake-model", timeout_seconds=7), client=TimeoutOllamaClient())

    with pytest.raises(OllamaGenerationTimeout, match="timed out after 7 seconds"):
        agent.answer(_task())


def test_ollama_model_missing_has_pull_message() -> None:
    agent = OllamaSeniorAgent(OllamaGenerationConfig("missing-model"), client=ModelListClient(["other-model"]))

    with pytest.raises(OllamaModelNotFound, match="ollama pull missing-model"):
        agent.check_model_available()


def _task():
    from tandem_rlvr.tasks.base import Task

    return Task("t1", "addition", "What is 2 + 3?", "5", "easy", {"left": 2, "right": 3, "operator": "+"})
