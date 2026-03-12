"""
aMOUR – encoders.music2vec
music2vec-v1 backend (data2vec architecture fine-tuned on music, 768-D).

Note: the processor comes from facebook/data2vec-audio-base-960h,
not from the music2vec model itself.
"""

import numpy as np
import torch
from transformers import Data2VecAudioModel, Wav2Vec2Processor

from encoders.base import AudioEncoder

MODEL_ID = "m-a-p/music2vec-v1"
PROCESSOR_ID = "facebook/data2vec-audio-base-960h"


class Music2VecEncoder(AudioEncoder):
    name = "music2vec-v1"
    sample_rate = 16_000
    embedding_dim = 768
    chunk_duration_s = 30

    def _load_model(self) -> None:
        print(f"Loading {MODEL_ID} …")
        self.processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_ID)
        self.model = Data2VecAudioModel.from_pretrained(MODEL_ID).to(self.device).eval()
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

            # last_hidden_state: (B, T, 768) → mean-pool time → (B, 768)
            pooled = out.last_hidden_state.mean(dim=1)
            all_pooled.append(pooled.cpu().float().numpy())

        stacked = np.concatenate(all_pooled, axis=0)   # (n_chunks, 768)
        return stacked.mean(axis=0)                     # (768,)
