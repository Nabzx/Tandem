from __future__ import annotations

from typing import Any

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.agents.llm.config import OllamaGenerationConfig
from tandem_rlvr.agents.llm.handoff_strategies import DEFAULT_HANDOFF_STRATEGY, get_handoff_strategy
from tandem_rlvr.agents.llm.parsing import normalize_llm_answer, parse_llm_response
from tandem_rlvr.agents.llm.prompts import (
    junior_direct_prompt,
    junior_tandem_prompt,
    senior_direct_prompt,
    senior_handoff_prompt,
)
from tandem_rlvr.tasks.base import Task


OLLAMA_UNAVAILABLE_MESSAGE = (
    "Ollama backend unavailable. Please install Ollama, run `ollama serve`, "
    "and pull a model such as `ollama pull llama3.2:1b`."
)


class OllamaBackendUnavailable(RuntimeError):
    """Raised when the Ollama Python client or local server is unavailable."""


class _BaseOllamaAgent:
    def __init__(
        self,
        config: OllamaGenerationConfig,
        client: Any | None = None,
        check_availability: bool = True,
    ) -> None:
        self.config = config
        self.model_name = config.model_name
        self._client = client if client is not None else self._make_client(config.timeout_seconds)
        if check_availability:
            self.check_available()

    def check_available(self) -> None:
        try:
            self._client.list()
        except Exception as exc:  # pragma: no cover - exact Ollama exceptions vary by version.
            raise OllamaBackendUnavailable(OLLAMA_UNAVAILABLE_MESSAGE) from exc

    def _complete(self, prompt: str, task: Task) -> AgentResponse:
        try:
            response = self._client.chat(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self.config.as_options(),
                format="json",
            )
        except Exception as exc:  # pragma: no cover - exact Ollama exceptions vary by version.
            raise OllamaBackendUnavailable(OLLAMA_UNAVAILABLE_MESSAGE) from exc

        raw_output = _extract_content(response)
        parsed = parse_llm_response(raw_output)
        final_answer = normalize_llm_answer(task, parsed.final_answer)
        return AgentResponse(
            reasoning=parsed.reasoning,
            final_answer=final_answer,
            metadata={
                **parsed.metadata,
                "model_name": self.config.model_name,
                "generation_config": self.config.as_dict(),
            },
        )

    @staticmethod
    def _make_client(timeout_seconds: int) -> Any:
        try:
            import ollama
        except ImportError as exc:
            raise OllamaBackendUnavailable(OLLAMA_UNAVAILABLE_MESSAGE) from exc
        return ollama.Client(timeout=timeout_seconds)


class OllamaSeniorAgent(_BaseOllamaAgent):
    name = "ollama_senior"

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        return self._complete(senior_direct_prompt(task), task)

    def produce_handoff(self, task: Task, strategy_name: str = DEFAULT_HANDOFF_STRATEGY) -> AgentResponse:
        strategy = get_handoff_strategy(strategy_name)
        return self._complete(
            senior_handoff_prompt(
                task,
                handoff_strategy_name=strategy.name,
                handoff_strategy_instruction=strategy.instruction,
            ),
            task,
        )


class OllamaJuniorAgent(_BaseOllamaAgent):
    name = "ollama_junior"

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        if context:
            return self._complete(junior_tandem_prompt(task, context), task)
        return self._complete(junior_direct_prompt(task), task)


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    content = getattr(message, "content", None)
    return "" if content is None else str(content)
