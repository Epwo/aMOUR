"""
aMOUR – encoders.mert
MERT-v1-330M (m-a-p, 2023) — music SSL model, 24 kHz, 1024-D, 24 layers.
Weights: CC-BY-NC 4.0.
"""

import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from encoders.base import MusicEncoder, pool_hidden_states


class MERTEncoder(MusicEncoder):
    name = "MERT-v1-330M"
    model_id = "m-a-p/MERT-v1-330M"
    sample_rate = 24_000
    embedding_dim = 1024
    chunk_duration_s = 30

    def _load_model(self) -> None:
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()

    def embed_batch(self, batch: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(batch, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)
        return pool_hidden_states(out.hidden_states, self.layers)
