"""
aMOUR – encode.py
Scans a directory of audio files, encodes each track with MERT on GPU,
and saves a compressed .npz archive with embeddings + metadata.

Usage:
    python src/encode.py --audio_dir data/ --output embeddings/embeddings.npz
    python src/encode.py --audio_dir data/ --output embeddings/embeddings.npz --batch_size 8
"""

import argparse
import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings("ignore", category=UserWarning)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
MERT_MODEL_ID = "m-a-p/MERT-v1-330M"
MERT_SAMPLE_RATE = 24_000  # MERT expects 24 kHz
CHUNK_DURATION_S = 30  # seconds per chunk — longer = richer embedding
OVERLAP_S = 5  # overlap between chunks (smooths boundary artefacts)


def load_audio(path: Path, target_sr: int = MERT_SAMPLE_RATE) -> np.ndarray | None:
    """Load and resample audio to mono @ target_sr. Returns None on failure."""
    try:
        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return audio
    except Exception as exc:
        print(f"  [WARN] Could not load {path.name}: {exc}")
        return None


def chunk_audio(
    audio: np.ndarray, sr: int, chunk_s: int, overlap_s: int
) -> list[np.ndarray]:
    """Split a waveform into overlapping chunks of chunk_s seconds."""
    chunk_len = chunk_s * sr
    hop_len = (chunk_s - overlap_s) * sr
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_len, len(audio))
        chunk = audio[start:end]
        # Pad last chunk if shorter than 1 s (avoid degenerate inputs)
        if len(chunk) < sr:
            break
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)
        if end == len(audio):
            break
        start += hop_len
    return chunks


def embed_chunks(
    chunks: list[np.ndarray],
    processor: AutoProcessor,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Run MERT on a list of audio chunks in batches.
    Returns a single (hidden_size,) vector = mean-pool across chunks and time.
    """
    all_chunk_embeddings = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        inputs = processor(
            batch,
            sampling_rate=MERT_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.last_hidden_state: (B, T, H)
        # Mean-pool over time dimension → (B, H)
        hidden = outputs.last_hidden_state  # last transformer layer
        pooled = hidden.mean(dim=1)  # (B, H)
        all_chunk_embeddings.append(pooled.cpu().float().numpy())

    # Stack all chunks then average → one vector per track
    stacked = np.concatenate(all_chunk_embeddings, axis=0)  # (n_chunks, H)
    return stacked.mean(axis=0)  # (H,)


def scan_audio_files(audio_dir: Path) -> list[Path]:
    files = sorted(
        p for p in audio_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def main():
    parser = argparse.ArgumentParser(description="aMOUR – MERT audio encoder")
    parser.add_argument(
        "--audio_dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing audio files (recursive scan)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embeddings/embeddings.npz"),
        help="Output .npz file path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of 30-s chunks per GPU batch (lower if OOM)",
    )
    parser.add_argument(
        "--chunk_s",
        type=int,
        default=CHUNK_DURATION_S,
        help="Chunk duration in seconds",
    )
    parser.add_argument(
        "--overlap_s",
        type=int,
        default=OVERLAP_S,
        help="Overlap between consecutive chunks in seconds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode files even if output already exists",
    )
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("No GPU found — running on CPU (this will be slow)")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading {MERT_MODEL_ID} …")
    processor = AutoProcessor.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
    model = model.to(device).eval()
    print(
        f"Model loaded  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)\n"
    )

    # ── Scan files ──────────────────────────────────────────────────────────
    audio_files = scan_audio_files(args.audio_dir)
    if not audio_files:
        print(
            f"No audio files found in {args.audio_dir}. "
            "Drop your .mp3/.wav/.flac files there and re-run."
        )
        return
    print(f"Found {len(audio_files)} audio file(s) in {args.audio_dir}\n")

    # ── Load existing embeddings (resume support) ───────────────────────────
    existing: dict[str, np.ndarray] = {}
    existing_meta: dict[str, str] = {}
    if args.output.exists() and not args.force:
        archive = np.load(args.output, allow_pickle=True)
        existing = {k: archive[k] for k in archive.files if not k.startswith("_meta_")}
        if "_meta_paths" in archive:
            paths_arr = archive["_meta_paths"]
            for i, name in enumerate(archive["_meta_names"]):
                existing_meta[str(name)] = str(paths_arr[i])
        print(f"Resuming — {len(existing)} track(s) already encoded.\n")

    # ── Encode ──────────────────────────────────────────────────────────────
    embeddings: dict[str, np.ndarray] = dict(existing)
    track_names: list[str] = list(existing_meta.keys()) if existing_meta else []
    track_paths: list[str] = list(existing_meta.values()) if existing_meta else []

    # Build lookup of already-done file stems
    done_keys = set(embeddings.keys())

    to_encode = [f for f in audio_files if f.stem not in done_keys]
    print(f"Encoding {len(to_encode)} new track(s) …\n")

    for audio_path in tqdm(to_encode, unit="track"):
        audio = load_audio(audio_path)
        if audio is None:
            continue

        chunks = chunk_audio(audio, MERT_SAMPLE_RATE, args.chunk_s, args.overlap_s)
        if not chunks:
            print(f"  [WARN] {audio_path.name} too short to chunk, skipping.")
            continue

        embedding = embed_chunks(chunks, processor, model, device, args.batch_size)

        key = audio_path.stem
        # Deduplicate keys (same stem, different folder)
        if key in embeddings:
            key = f"{audio_path.parent.name}__{audio_path.stem}"
        embeddings[key] = embedding
        track_names.append(key)
        track_paths.append(str(audio_path.relative_to(args.audio_dir)))

    # ── Save ────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **embeddings,
        _meta_names=np.array(track_names),
        _meta_paths=np.array(track_paths),
    )
    print(f"\nSaved {len(embeddings)} embeddings → {args.output}")
    print(f"Embedding dimension: {next(iter(embeddings.values())).shape[0]}")


if __name__ == "__main__":
    main()
