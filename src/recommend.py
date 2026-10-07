"""
aMOUR – recommend.py
Rank tracks by cosine similarity in the original embedding space.

Three kinds of query:
  • a track already in the index (exact / substring / fuzzy name match)
  • a NEW audio file, encoded on the fly with the archive's encoder
  • a text description (joint music–text encoders only: mulan, clap)

Usage:
    python src/recommend.py "bohemian"                              # single archive in embeddings/
    python src/recommend.py "bohemian" --encoder muq --top_k 20
    python src/recommend.py --file path/to/new_song.mp3 --encoder muq
    python src/recommend.py --text "melancholic piano, rainy night" --encoder mulan
    python src/recommend.py --list --input embeddings/muq_deezer.npz
"""

import argparse
from pathlib import Path

from common import (
    Archive, chunk_audio, get_device, load_archive, load_audio,
    resolve_archive_path, resolve_query, suggest_names,
)
from similarity import query_by_index, query_by_vector


def load_matching_encoder(archive: Archive, encoder_arg: str | None):
    """Instantiate the encoder that produced the archive (vectors must share a space)."""
    from encoders import AVAILABLE_ENCODERS, get_encoder

    key = encoder_arg or next(
        (k for k, cls in AVAILABLE_ENCODERS.items() if cls.name == archive.encoder), None
    )
    if key is None:
        raise SystemExit(f"[ERROR] Unknown encoder '{archive.encoder}' in archive; pass --encoder.")
    encoder = get_encoder(key, device=get_device(verbose=False))
    if encoder.name != archive.encoder:
        raise SystemExit(f"[ERROR] Archive was made with {archive.encoder}, not {encoder.name}.")
    return encoder


def print_results(title: str, results: list[tuple[int, float]], archive: Archive) -> None:
    sep = "═" * 64
    print(f"\n{sep}\n  aMOUR — {title}\n  Encoder: {archive.encoder}  |  dim: {archive.dim}\n{sep}")
    for rank, (j, score) in enumerate(results, 1):
        bar_len = max(0, round(score * 30))
        print(
            f"\n  {rank:>2}.  {archive.names[j]}"
            f"\n       {'█' * bar_len}{'░' * (30 - bar_len)}  sim={score:.4f}  [{archive.labels[j]}]"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="aMOUR – music recommendations")
    parser.add_argument("query", nargs="?", help="track name (or part of it) in the index")
    parser.add_argument("--file", type=Path, help="new audio file to encode and match")
    parser.add_argument("--text", help="text description to match (mulan / clap archives)")
    parser.add_argument("--encoder", help="picks embeddings/<encoder>_*.npz when --input is omitted")
    parser.add_argument("--input", type=Path, help="embeddings archive (.npz)")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--list", action="store_true", dest="list_tracks", help="list indexed tracks")
    args = parser.parse_args()

    archive = load_archive(resolve_archive_path(args.input, args.encoder))
    print(f"Loaded {len(archive)} tracks from {archive.path}  |  {archive.encoder}")

    if args.list_tracks:
        for i, (n, lbl) in enumerate(zip(archive.names, archive.labels)):
            print(f"  {i:>4}. {n:<50s}  [{lbl}]")
        return

    if args.text:
        encoder = load_matching_encoder(archive, args.encoder)
        if not encoder.supports_text:
            raise SystemExit(f"[ERROR] {encoder.name} has no text tower — use a mulan or clap archive.")
        query_vec = encoder.embed_text([args.text])[0]
        print_results(f'"{args.text}"', query_by_vector(query_vec, archive.matrix, args.top_k), archive)
        return

    if args.file:
        if not args.file.exists():
            raise SystemExit(f"[ERROR] File not found: {args.file}")
        encoder = load_matching_encoder(archive, args.encoder)
        audio = load_audio(args.file, encoder.sample_rate)
        chunks = chunk_audio(audio, encoder.sample_rate, encoder.chunk_duration_s) if audio is not None else []
        if not chunks:
            raise SystemExit("[ERROR] Could not load audio or it is too short.")
        query_vec = encoder.encode_chunks(chunks)
        print_results(args.file.name, query_by_vector(query_vec, archive.matrix, args.top_k), archive)
        return

    if args.query is None:
        parser.print_help()
        return

    idx = resolve_query(args.query, archive.names)
    if idx is None:
        print(f"\n[ERROR] No track matching '{args.query}' (use --list).")
        for s in suggest_names(args.query, archive.names):
            print(f"  • {s}")
        return
    print_results(archive.names[idx], query_by_index(idx, archive.matrix, args.top_k), archive)


if __name__ == "__main__":
    main()
