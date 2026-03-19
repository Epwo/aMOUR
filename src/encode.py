"""
aMOUR – encode.py
Scans a directory of audio files, encodes each track with a swappable encoder
backend, and saves a compressed .npz archive with embeddings + metadata.

Usage:
    python src/encode.py --audio_dir data/                           # MERT (default)
    python src/encode.py --audio_dir data/ --encoder clap            # CLAP
    python src/encode.py --audio_dir data/ --encoder music2vec       # music2vec
    python src/encode.py --audio_dir data/ --batch_size 8            # larger GPU batches
    python src/encode.py --audio_dir data/ --encoder clap --force    # re-encode everything

Each encoder + dataset combo gets its own output file:
    embeddings/mert_deezer.npz / embeddings/clap_deezer.npz / …
"""

import argparse
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

from encoders import AVAILABLE_ENCODERS, get_encoder

warnings.filterwarnings("ignore", category=UserWarning)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
OVERLAP_S = 5


def load_audio(path: Path, target_sr: int) -> np.ndarray | None:
    """Load and resample audio to mono @ target_sr. Returns None on failure."""
    try:
        audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
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
        if len(chunk) < sr:
            break
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)
        if end == len(audio):
            break
        start += hop_len
    return chunks


def scan_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(
        p for p in audio_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    encoder_names = ", ".join(AVAILABLE_ENCODERS.keys())

    parser = argparse.ArgumentParser(description="aMOUR – audio encoder")
    parser.add_argument(
        "--encoder", type=str, default="mert",
        help=f"Encoder backend to use ({encoder_names})",
    )
    parser.add_argument(
        "--audio_dir", type=Path, default=Path("data"),
        help="Root directory containing audio files (recursive scan)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .npz file path (default: embeddings/<encoder>.npz)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of chunks per GPU batch (lower if OOM)",
    )
    parser.add_argument(
        "--overlap_s", type=int, default=OVERLAP_S,
        help="Overlap between consecutive chunks in seconds",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-encode files even if output already exists",
    )
    args = parser.parse_args()

    # Default output: embeddings/<encoder>_<dataset>.npz
    if args.output is None:
        dataset_name = args.audio_dir.name  # e.g. "deezer", "fma_small"
        args.output = Path(f"embeddings/{args.encoder.lower()}_{dataset_name}.npz")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("No GPU found — running on CPU (this will be slow)")

    # ── Load encoder ────────────────────────────────────────────────────────
    encoder = get_encoder(args.encoder, device=device)
    print(f"\n{encoder}\n")

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
    done_keys = set(embeddings.keys())

    to_encode = [f for f in audio_files if f.stem not in done_keys]
    print(f"Encoding {len(to_encode)} new track(s) with {encoder.name} …\n")

    for audio_path in tqdm(to_encode, unit="track"):
        audio = load_audio(audio_path, target_sr=encoder.sample_rate)
        if audio is None:
            continue

        chunks = chunk_audio(
            audio, encoder.sample_rate, encoder.chunk_duration_s, args.overlap_s,
        )
        if not chunks:
            print(f"  [WARN] {audio_path.name} too short to chunk, skipping.")
            continue

        embedding = encoder.encode_chunks(chunks, batch_size=args.batch_size)

        key = audio_path.stem
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
        _meta_encoder=np.array(encoder.name),
    )
    print(f"\nSaved {len(embeddings)} embeddings → {args.output}")
    print(f"Encoder: {encoder.name}  |  Embedding dim: {encoder.embedding_dim}")


if __name__ == "__main__":
    main()
