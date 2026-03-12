"""
aMOUR – encoders.clap
CLAP (larger_clap_music) backend — contrastive audio-text model, 512-D embeddings.

Note: CLAP's processor truncates/pads audio to a fixed window (~10 s).
We chunk and mean-pool to handle full-length tracks.
"""

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from encoders.base import AudioEncoder

MODEL_ID = "laion/larger_clap_music"


class CLAPEncoder(AudioEncoder):
    name = "CLAP-music"
    sample_rate = 48_000
    embedding_dim = 512
    chunk_duration_s = 10  # CLAP's native window

    def _load_model(self) -> None:
        print(f"Loading {MODEL_ID} …")
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)
        self.model = ClapModel.from_pretrained(MODEL_ID).to(self.device).eval()
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
                audios=batch,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # get_audio_features → (B, 512), already projected
                embeddings = self.model.get_audio_features(**inputs)

            all_pooled.append(embeddings.cpu().float().numpy())

        stacked = np.concatenate(all_pooled, axis=0)   # (n_chunks, 512)
        return stacked.mean(axis=0)                     # (512,)
