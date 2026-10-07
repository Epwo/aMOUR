"""
aMOUR – encode.py
Encode every audio file under a directory with a music encoder and save
an .npz archive (embeddings matrix + names, paths, labels).

Usage:
    python src/encode.py --audio_dir data/deezer                  # MuQ (default)
    python src/encode.py --audio_dir data/deezer --encoder mulan  # MuQ-MuLan (enables --text search)
    python src/encode.py --audio_dir data/fma_small --limit 800   # quick benchmark subset
    python src/encode.py --audio_dir data/deezer --layers 4-8     # pool only some layers
    python src/encode.py --audio_dir data/deezer --force          # re-encode everything

Output: embeddings/<encoder>_<dataset>.npz (resume-safe: already-encoded
tracks are skipped on re-run).
"""

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    Archive, chunk_audio, default_archive_path, get_device, infer_labels,
    load_archive, load_audio, save_archive, scan_audio_files,
)
from encoders import AVAILABLE_ENCODERS, canonical_name, get_encoder

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    parser = argparse.ArgumentParser(description="aMOUR – music encoder")
    parser.add_argument("--encoder", default="muq", help=f"one of: {', '.join(AVAILABLE_ENCODERS)}")
    parser.add_argument("--audio_dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None,
                        help="default: embeddings/<encoder>_<dataset>.npz")
    parser.add_argument("--layers", default=None,
                        help="hidden layers to pool for SSL models, e.g. '6', '4-8' (default: all)")
    parser.add_argument("--batch_size", type=int, default=4, help="chunks per forward pass (lower if OOM)")
    parser.add_argument("--overlap_s", type=float, default=5.0, help="overlap between chunks (s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="encode a random sample of N tracks (seeded) — handy for FMA benchmarks")
    parser.add_argument("--force", action="store_true", help="ignore any existing archive")
    args = parser.parse_args()

    key = canonical_name(args.encoder)
    output = args.output or default_archive_path(key, args.audio_dir)

    audio_files = scan_audio_files(args.audio_dir)
    if not audio_files:
        raise SystemExit(f"No audio files found in {args.audio_dir}.")
    if args.limit and args.limit < len(audio_files):
        audio_files = sorted(random.Random(0).sample(audio_files, args.limit))
    print(f"Found {len(audio_files)} audio file(s) in {args.audio_dir}")

    # ── Resume ────────────────────────────────────────────────────────────────
    vectors: list[np.ndarray] = []
    names: list[str] = []
    paths: list[str] = []
    if output.exists() and not args.force:
        prev = load_archive(output)
        if prev.legacy:
            print(f"{output} uses the old format/pooling — re-encoding from scratch.")
        else:
            vectors, names, paths = list(prev.matrix), prev.names, prev.paths
            print(f"Resuming — {len(names)} track(s) already in {output}")

    done = set(paths)
    todo = [f for f in audio_files if f.relative_to(args.audio_dir).as_posix() not in done]
    if not todo:
        print("Nothing to encode.")
        return

    encoder = get_encoder(key, device=get_device(), layers=args.layers)
    print(f"{encoder}\n\nEncoding {len(todo)} track(s) …")

    for audio_path in tqdm(todo, unit="track"):
        audio = load_audio(audio_path, encoder.sample_rate)
        if audio is None:
            continue
        chunks = chunk_audio(audio, encoder.sample_rate, encoder.chunk_duration_s, args.overlap_s)
        if not chunks:
            print(f"  [WARN] {audio_path.name} too short, skipping.")
            continue
        vectors.append(encoder.encode_chunks(chunks, batch_size=args.batch_size))
        names.append(audio_path.stem)
        paths.append(audio_path.relative_to(args.audio_dir).as_posix())

    # Labels are recomputed for every track so metadata fixes apply on resume.
    labels = infer_labels([args.audio_dir / p for p in paths], args.audio_dir)
    save_archive(output, Archive(np.stack(vectors), names, paths, labels, encoder.name, output))
    print(f"\nSaved {len(names)} embeddings ({encoder.embedding_dim}-D) → {output}")


if __name__ == "__main__":
    main()
