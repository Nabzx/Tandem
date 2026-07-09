from __future__ import annotations

import time
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


class OllamaBackendError(RuntimeError):
    """Base error for local Ollama backend failures."""

    failure_type = "backend_error"


class OllamaBackendUnavailable(OllamaBackendError):
    """Raised when the Ollama Python client or local server is unavailable."""

    failure_type = "backend_unavailable"


class OllamaModelNotFound(OllamaBackendError):
    """Raised when Ollama is reachable but the requested model is missing."""

    failure_type = "model_not_found"


class OllamaGenerationTimeout(OllamaBackendError):
    """Raised when a local generation exceeds the configured timeout."""

    failure_type = "timeout"


class OllamaMalformedResponse(OllamaBackendError):
    """Raised when Ollama responds without usable message content."""

    failure_type = "backend_error"


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
            _raise_ollama_error(exc, self.config, operation="list")

    def check_model_available(self) -> None:
        try:
            response = self._client.list()
        except Exception as exc:  # pragma: no cover - exact Ollama exceptions vary by version.
            _raise_ollama_error(exc, self.config, operation="list")
        model_names = _extract_model_names(response)
        if self.config.model_name not in model_names:
            raise OllamaModelNotFound(_model_not_found_message(self.config.model_name))

    def warmup(self) -> float:
        start = time.monotonic()
        try:
            response = self._client.chat(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": 'Return only JSON: {"reasoning":"warmup","final_answer":"ok"}',
                    }
                ],
                options={**self.config.as_options(), "num_predict": min(16, int(self.config.num_predict))},
                format="json",
            )
        except Exception as exc:  # pragma: no cover - exact Ollama exceptions vary by version.
            _raise_ollama_error(exc, self.config, operation="chat")
        raw_output = _extract_content(response)
        if not raw_output.strip():
            raise OllamaMalformedResponse(f"Ollama returned a malformed response for model {self.config.model_name}.")
        return time.monotonic() - start

    def _complete(self, prompt: str, task: Task) -> AgentResponse:
        try:
            response = self._client.chat(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self.config.as_options(),
                format="json",
            )
        except Exception as exc:  # pragma: no cover - exact Ollama exceptions vary by version.
            _raise_ollama_error(exc, self.config, operation="chat")

        raw_output = _extract_content(response)
        if not raw_output.strip():
            raise OllamaMalformedResponse(f"Ollama returned a malformed response for model {self.config.model_name}.")
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


def _raise_ollama_error(exc: Exception, config: OllamaGenerationConfig, operation: str) -> None:
    if _is_timeout_error(exc):
        raise OllamaGenerationTimeout(_timeout_message(config.timeout_seconds)) from exc
    if _is_model_not_found_error(exc):
        raise OllamaModelNotFound(_model_not_found_message(config.model_name)) from exc
    if _is_backend_unavailable_error(exc):
        raise OllamaBackendUnavailable(OLLAMA_UNAVAILABLE_MESSAGE) from exc
    raise OllamaBackendError(f"Ollama backend error while running {operation} for model {config.model_name}: {exc}") from exc


def _is_timeout_error(exc: Exception) -> bool:
    text = _exception_text(exc)
    return "timeout" in text or "timed out" in text


def _is_model_not_found_error(exc: Exception) -> bool:
    text = _exception_text(exc)
    status_code = getattr(exc, "status_code", None)
    return status_code == 404 or "model" in text and "not found" in text


def _is_backend_unavailable_error(exc: Exception) -> bool:
    text = _exception_text(exc)
    return any(
        marker in text
        for marker in (
            "connection refused",
            "connecterror",
            "connection error",
            "failed to connect",
            "not running",
            "no such file",
        )
    )


def _exception_text(exc: Exception) -> str:
    return f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}".lower()


def _timeout_message(timeout_seconds: int) -> str:
    return (
        f"Ollama generation timed out after {timeout_seconds} seconds. "
        "Try increasing --max-generation-seconds, reducing --num-predict, using --quick, "
        "or warming the model with `ollama run <model>`."
    )


def _model_not_found_message(model_name: str) -> str:
    return f"Model not found. Run: ollama pull {model_name}"


def _extract_model_names(response: Any) -> set[str]:
    if isinstance(response, dict):
        models = response.get("models", [])
    else:
        models = getattr(response, "models", [])
    names: set[str] = set()
    for model in models or []:
        if isinstance(model, dict):
            for key in ("name", "model"):
                value = model.get(key)
                if value:
                    names.add(str(value))
        else:
            for key in ("name", "model"):
                value = getattr(model, key, None)
                if value:
                    names.add(str(value))
    return names
