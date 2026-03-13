"""
aMOUR – encoders.encodec
EnCodec 48kHz backend — Facebook's neural audio codec.

Uses the encoder output BEFORE the residual vector quantizer,
giving continuous 128-D embeddings (not discrete codes).

This is the encoder used inside MusicGen. Useful as a baseline
to compare against semantic models like MERT/CLAP.

Note: EnCodec is optimised for reconstruction fidelity, not
semantic similarity — so we expect weaker genre clustering
compared to MERT or CLAP, but it's interesting to verify.
"""

import numpy as np
import torch
from transformers import EncodecModel, AutoProcessor

from encoders.base import AudioEncoder

MODEL_ID = "facebook/encodec_48khz"


class EnCodecEncoder(AudioEncoder):
    name = "EnCodec-48kHz"
    sample_rate = 48_000
    embedding_dim = 128
    chunk_duration_s = 15  # we chunk at a higher level; internal chunking is 1s

    def _load_model(self) -> None:
        print(f"Loading {MODEL_ID} …")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = EncodecModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self._config = self.model.config
        print(f"  {self.name} ready ({self.param_count() / 1e6:.0f}M params)")

    @torch.no_grad()
    def _encode_single(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode a single audio chunk → (128,) vector.
        Handles EnCodec's internal 1-s sub-chunking and normalisation.
        """
        inputs = self.processor(
            raw_audio=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)  # (1, 2, T)

        config = self._config
        sr = config.sampling_rate
        chunk_len_s = config.chunk_length_s
        all_embs = []

        if chunk_len_s is not None:
            chunk_len = int(chunk_len_s * sr)
            stride = int(chunk_len * (1 - config.overlap))
            length = input_values.shape[-1]

            for offset in range(0, length, stride):
                chunk = input_values[..., offset: offset + chunk_len]

                # Pad last sub-chunk
                if chunk.shape[-1] < chunk_len:
                    pad = chunk_len - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad))

                # Normalise (same as _encode_frame when config.normalize=True)
                if config.normalize:
                    mono = chunk.mean(dim=1, keepdim=True)
                    scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                    chunk = chunk / scale

                emb = self.model.encoder(chunk)  # (1, 128, frames)
                all_embs.append(emb)
        else:
            if config.normalize:
                mono = input_values.mean(dim=1, keepdim=True)
                scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                input_values = input_values / scale
            emb = self.model.encoder(input_values)
            all_embs.append(emb)

        # Concat time frames, mean-pool → (128,)
        combined = torch.cat(all_embs, dim=-1)  # (1, 128, total_frames)
        pooled = combined.mean(dim=-1).squeeze(0)  # (128,)
        return pooled.cpu().float().numpy()

    def encode_chunks(
        self,
        chunks: list[np.ndarray],
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        EnCodec handles its own internal sub-chunking, so we process
        our higher-level chunks one at a time and average.
        """
        all_embeddings = []
        for chunk in chunks:
            emb = self._encode_single(chunk)
            all_embeddings.append(emb)

        stacked = np.stack(all_embeddings)   # (n_chunks, 128)
        return stacked.mean(axis=0)          # (128,)
