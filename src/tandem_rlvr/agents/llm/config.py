from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OllamaGenerationConfig:
    model_name: str
    temperature: float = 0.0
    top_p: float = 1.0
    num_predict: int = 256
    timeout_seconds: int = 120

    def as_options(self) -> dict[str, float | int]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
        }

    def as_dict(self) -> dict[str, str | float | int]:
        return asdict(self)
