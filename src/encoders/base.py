"""
aMOUR – encoders.base
Abstract base class that every encoder backend must implement.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class AudioEncoder(ABC):
    """
    Contract for an audio encoder backend.

    Subclasses must implement:
        _load_model()        – download / load weights + processor
        encode_chunks()      – list[np.ndarray] → (hidden_size,) vector
        sample_rate          – expected input sample rate (Hz)
        embedding_dim        – dimensionality of the output vector
        name                 – human-readable identifier
        chunk_duration_s     – recommended chunk length in seconds
    """

    def __init__(self, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._load_model()

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable encoder name (e.g. 'MERT-v1-330M')."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate expected by this encoder (Hz)."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""

    @property
    @abstractmethod
    def chunk_duration_s(self) -> int:
        """Recommended chunk length in seconds."""

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights and processor onto self.device."""

    @abstractmethod
    def encode_chunks(
        self,
        chunks: list[np.ndarray],
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        Encode a list of audio chunks into a single embedding vector.

        Parameters
        ----------
        chunks     : list of 1-D float32 arrays (mono, at self.sample_rate)
        batch_size : how many chunks to process per GPU forward pass

        Returns
        -------
        embedding : (self.embedding_dim,) float32 array
                    Mean-pooled across all chunks.
        """

    # ── Shared helpers ────────────────────────────────────────────────────────

    def param_count(self) -> int:
        """Total number of model parameters."""
        if hasattr(self, "model"):
            return sum(p.numel() for p in self.model.parameters())
        return 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, sr={self.sample_rate}, "
            f"dim={self.embedding_dim}, device={self.device})"
        )
