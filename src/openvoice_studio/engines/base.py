"""Engine adapter contract.

An adapter turns (text chunk, prepared voice) into audio. Everything else — chunking, pauses,
loudness, artifacts, evaluation — is shared, so engines can be benchmarked with the same protocol
(see CONTRIBUTING.md). Adapters must be safe to call from a single worker thread; they are never
called concurrently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EngineUnavailable(RuntimeError):
    pass


class Engine(ABC):
    name: str = ""
    label: str = ""
    license: str = ""
    sample_rate: int = 24000
    languages: dict[str, str] = {}
    experimental: bool = False

    @abstractmethod
    def status(self) -> dict:
        """Cheap check (no model load): {"available": bool, "models_installed": bool, "detail": str}."""

    @abstractmethod
    def load(self) -> None:
        """Load weights (idempotent). May download models on first use."""

    @abstractmethod
    def prepare_voice(self, voice_dir: Path, reference_wav: Path, samples_wav: Path) -> None:
        """Compute and cache whatever the engine needs to speak with this voice."""

    @abstractmethod
    def synthesize(self, text: str, voice_dir: Path, language: str, params: dict) -> np.ndarray:
        """Return float32 mono audio at self.sample_rate for one chunk of text."""

    @abstractmethod
    def speaker_embedding(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """L2-normalised speaker embedding, used for the similarity metric."""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "license": self.license,
            "sample_rate": self.sample_rate,
            "languages": self.languages,
            "experimental": self.experimental,
            **self.status(),
        }
