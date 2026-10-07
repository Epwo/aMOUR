"""
aMOUR – encoders.omar
OMAR-RQ multicodebook (MTG-UPF, 2025) — `pip install omar-rq`.
Music SSL model trained on 330k h of music, 16 kHz, conformer encoder.
The multicodebook variant is the one recommended for tagging/semantic tasks.
Weights: CC-BY-NC-SA 4.0.
"""

import numpy as np
import torch

from encoders.base import MusicEncoder


class OMARRQEncoder(MusicEncoder):
    name = "OMAR-RQ-multicodebook"
    model_id = "mtg-upf/omar-rq-multicodebook"
    sample_rate = 16_000
    embedding_dim = 1024
    chunk_duration_s = 30  # model's max input length

    def _load_model(self) -> None:
        from omar_rq import get_model

        self.model = get_model(model_id=self.model_id, device=str(self.device)).eval()
        n_layers = len(self.model.net.layers)
        self._layer_set = set(self.layers) if self.layers is not None else set(range(n_layers))

    def embed_batch(self, batch: list[np.ndarray]) -> torch.Tensor:
        emb = self.model.extract_embeddings(self._to_tensor(batch), layers=self._layer_set)
        return emb.mean(dim=2).mean(dim=0)  # (L, B, T, C) → (B, C)
