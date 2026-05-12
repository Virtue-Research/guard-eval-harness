"""Base model adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from guard_eval_harness.config.models import ResolvedModelConfig
from guard_eval_harness.schemas import (
    AdapterCapabilities,
    NormalizedPrediction,
    PredictSample,
)


class ModelAdapter(ABC):
    """Base interface for model backends."""

    def __init__(self, config: ResolvedModelConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: ResolvedModelConfig) -> "ModelAdapter":
        """Build an adapter from resolved config."""
        return cls(config)

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Return adapter capability flags."""

    @property
    def allow_partial_predictions(self) -> bool:
        """Whether the runner may drop missing predictions for this adapter."""
        return False

    @abstractmethod
    def predict_batch(
        self,
        samples: Sequence[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        """Emit canonical predictions for a batch of samples."""


class PlaceholderAdapter(ModelAdapter):
    """Base class for scaffolded but not yet implemented adapters."""

    capability_notes: tuple[str, ...] = ("scaffold-only",)
    adapter_name: str = "placeholder"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            adapter_name=self.adapter_name,
            probability_scores=False,
            batching=False,
            concurrency=False,
            cost_estimation=False,
            token_accounting=False,
            notes=self.capability_notes,
        )

    def predict_batch(
        self,
        samples: Sequence[PredictSample],
        *,
        threshold: float,
    ) -> list[NormalizedPrediction]:
        raise NotImplementedError(
            f"{self.adapter_name} is scaffolded but not implemented yet"
        )
