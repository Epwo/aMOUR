"""
aMOUR – encoders.muq
MuQ family (Tencent AI Lab, 2025) — `pip install muq`. Weights: CC-BY-NC 4.0.

MuQ        : music SSL model (Mel-RVQ targets), 24 kHz, 1024-D, 12 layers.
             Beats MERT on almost every MARBLE task with far less training data.
MuQ-MuLan  : MuQ fine-tuned contrastively against text, 512-D joint space.
             Best off-the-shelf model in recent perceptual music-similarity
             studies (arXiv:2601.19109) and lets you search music with text.
"""

import numpy as np
import torch

from encoders.base import MusicEncoder, pool_hidden_states


class MuQEncoder(MusicEncoder):
    name = "MuQ-large-msd"
    model_id = "OpenMuQ/MuQ-large-msd-iter"
    sample_rate = 24_000
    embedding_dim = 1024
    chunk_duration_s = 30

    def _load_model(self) -> None:
        from muq import MuQ

        self.model = MuQ.from_pretrained(self.model_id).to(self.device).eval()

    def embed_batch(self, batch: list[np.ndarray]) -> torch.Tensor:
        out = self.model(self._to_tensor(batch), output_hidden_states=True)
        return pool_hidden_states(out.hidden_states, self.layers)


class MuQMuLanEncoder(MusicEncoder):
    name = "MuQ-MuLan-large"
    model_id = "OpenMuQ/MuQ-MuLan-large"
    sample_rate = 24_000
    embedding_dim = 512
    chunk_duration_s = 10  # MuLan's native clip length
    supports_text = True

    def _load_model(self) -> None:
        from muq import MuQMuLan

        self.model = MuQMuLan.from_pretrained(self.model_id).to(self.device).eval()

    def embed_batch(self, batch: list[np.ndarray]) -> torch.Tensor:
        # Feed our 10 s chunks straight to the audio tower (the high-level
        # wrapper would re-chunk and loop one clip at a time).
        return self.model.mulan_module.get_audio_latents(self._to_tensor(batch))

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        return self.model(texts=texts).float().cpu().numpy()
