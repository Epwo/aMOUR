"""
aMOUR – encoders.base
Contract every music encoder backend implements.

A backend only has to turn a batch of equal-length mono chunks into one
vector per chunk (`embed_batch`). Batching over a track's chunks and
pooling them into a single track vector is shared here.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


def parse_layers(spec: str | None) -> list[int] | None:
    """'6' → [6], '4-8' → [4..8], '3,6,9' → [3, 6, 9], None/'all' → None (all)."""
    if spec is None or spec == "all":
        return None
    layers: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    return layers


def pool_hidden_states(hidden_states, layers: list[int] | None) -> torch.Tensor:
    """
    Time-average each selected transformer layer, then average the layers.
    hidden_states: sequence of (B, T, D) with index 0 = pre-transformer features.
    Different layers capture different things (timbre low, harmony/genre mid,
    pretext-task specifics top), so averaging several is a robust default.
    """
    idx = layers if layers is not None else range(1, len(hidden_states))
    return torch.stack([hidden_states[i].mean(dim=1) for i in idx]).mean(dim=0)


class MusicEncoder(ABC):
    name: str               # human-readable, stored in archives
    model_id: str           # Hugging Face id
    sample_rate: int        # expected input rate (Hz)
    embedding_dim: int      # output vector size
    chunk_duration_s: float  # window length fed to the model
    supports_text = False    # True for joint music–text models (text → music search)

    def __init__(self, device: torch.device | None = None, layers: str | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = parse_layers(layers)
        print(f"Loading {self.model_id} …")
        self._load_model()
        print(f"  {self.name} ready ({self.param_count() / 1e6:.0f}M params)")

    @abstractmethod
    def _load_model(self) -> None:
        """Load weights onto self.device and set self.model."""

    @abstractmethod
    def embed_batch(self, batch: list[np.ndarray]) -> torch.Tensor:
        """Equal-length mono chunks @ sample_rate → (B, embedding_dim)."""

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Texts → (N, embedding_dim), in the same space as audio embeddings."""
        raise NotImplementedError(f"{self.name} has no text tower")

    @torch.no_grad()
    def encode_chunks(self, chunks: list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        """All chunks of one track → a single mean-pooled (embedding_dim,) vector."""
        # Chunks are equal length except a lone short one, so batching is safe.
        vecs = [
            self.embed_batch(chunks[i : i + batch_size]).float().cpu()
            for i in range(0, len(chunks), batch_size)
        ]
        return torch.cat(vecs).mean(dim=0).numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _to_tensor(self, batch: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(batch)).to(self.device)

    def __repr__(self) -> str:
        layers = "all" if self.layers is None else self.layers
        return (
            f"{self.__class__.__name__}(name={self.name!r}, sr={self.sample_rate}, "
            f"dim={self.embedding_dim}, layers={layers}, device={self.device})"
        )
