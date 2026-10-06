"""
aMOUR – encoders.clap
LAION CLAP (larger_clap_music) — contrastive audio–text model, 48 kHz, 512-D.
Kept as the general-audio baseline to measure the music models against.
"""

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from encoders.base import MusicEncoder


class CLAPEncoder(MusicEncoder):
    name = "CLAP-music"
    model_id = "laion/larger_clap_music"
    sample_rate = 48_000
    embedding_dim = 512
    chunk_duration_s = 10  # CLAP's native window
    supports_text = True

    def _load_model(self) -> None:
        self.processor = ClapProcessor.from_pretrained(self.model_id)
        self.model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()

    def embed_batch(self, batch: list[np.ndarray]) -> torch.Tensor:
        inputs = self.processor(audio=batch, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_audio_features(**inputs)

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs).float().cpu().numpy()
