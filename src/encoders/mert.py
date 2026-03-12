"""
aMOUR – encoders.mert
MERT-v1-330M backend (music understanding, 1024-D embeddings).
"""

import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from encoders.base import AudioEncoder

MODEL_ID = "m-a-p/MERT-v1-330M"


class MERTEncoder(AudioEncoder):
    name = "MERT-v1-330M"
    sample_rate = 24_000
    embedding_dim = 1024
    chunk_duration_s = 30

    def _load_model(self) -> None:
        print(f"Loading {MODEL_ID} …")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_ID, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True,
        ).to(self.device).eval()
        print(f"  {self.name} ready ({self.param_count() / 1e6:.0f}M params)")

    def encode_chunks(
        self,
        chunks: list[np.ndarray],
        batch_size: int = 4,
    ) -> np.ndarray:
        all_pooled = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            inputs = self.processor(
                batch,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)

            # last_hidden_state: (B, T, 1024) → mean-pool time → (B, 1024)
            pooled = out.last_hidden_state.mean(dim=1)
            all_pooled.append(pooled.cpu().float().numpy())

        stacked = np.concatenate(all_pooled, axis=0)   # (n_chunks, 1024)
        return stacked.mean(axis=0)                     # (1024,)
