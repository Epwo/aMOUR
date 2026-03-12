"""
aMOUR – recommend.py
Given a query track, returns the most similar songs ranked by cosine
similarity in the original high-dimensional embedding space.

Supports:
  • Exact track name match
  • Substring / fuzzy search (picks the best match)
  • Encoding a NEW audio file on-the-fly and comparing against the index

Usage:
    python src/recommend.py "Bohemian Rhapsody"
    python src/recommend.py "bohemian"                          # fuzzy
    python src/recommend.py "bohemian" --top_k 10
    python src/recommend.py --file path/to/new_song.mp3         # encode + recommend
    python src/recommend.py "bohemian" --input embeddings/clap.npz
"""

import argparse
from difflib import get_close_matches
from pathlib import Path

import numpy as np

from similarity import cosine_sim_matrix


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_embeddings(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load the .npz archive and return (matrix, names, paths)."""
    archive = np.load(path, allow_pickle=True)
    keys = [k for k in archive.files if not k.startswith("_meta_")]
    if not keys:
        raise ValueError(f"No embeddings found in {path}")

    matrix = np.stack([archive[k] for k in keys])
    names = (
        [str(n) for n in archive["_meta_names"]]
        if "_meta_names" in archive.files
        else keys
    )
    paths = (
        [str(p) for p in archive["_meta_paths"]]
        if "_meta_paths" in archive.files
        else keys
    )
    encoder = (
        str(archive["_meta_encoder"])
        if "_meta_encoder" in archive.files
        else "unknown"
    )
    return matrix, names, paths, encoder


def resolve_query(query: str, names: list[str]) -> int | None:
    """
    Resolve a user query to a track index.
    Tries: exact match → case-insensitive → substring → fuzzy.
    Returns the index or None.
    """
    # Exact match
    if query in names:
        return names.index(query)

    # Case-insensitive
    lower_names = [n.lower() for n in names]
    if query.lower() in lower_names:
        return lower_names.index(query.lower())

    # Substring (first match)
    ql = query.lower()
    for i, n in enumerate(lower_names):
        if ql in n:
            return i

    # Fuzzy (difflib)
    matches = get_close_matches(query.lower(), lower_names, n=1, cutoff=0.4)
    if matches:
        return lower_names.index(matches[0])

    return None


def recommend_for_index(
    query_idx: int,
    matrix: np.ndarray,
    names: list[str],
    paths: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    Return top_k most similar tracks for a given track index.

    Each result is a dict with keys: rank, name, path, similarity, distance.
    """
    sim = cosine_sim_matrix(matrix)  # (N, N)
    row = sim[query_idx].copy()
    row[query_idx] = -np.inf  # exclude self

    k = min(top_k, len(names) - 1)
    top_idx = np.argpartition(row, -k)[-k:]
    top_idx = top_idx[np.argsort(row[top_idx])[::-1]]

    results = []
    for rank, j in enumerate(top_idx, 1):
        results.append({
            "rank": rank,
            "name": names[j],
            "path": paths[j],
            "similarity": float(row[j]),
            "distance": float(1.0 - row[j]),
        })
    return results


def recommend_for_vector(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    names: list[str],
    paths: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    Return top_k most similar tracks for an arbitrary embedding vector
    (e.g. a newly encoded song not in the index).
    """
    # Cosine similarity: query vs all
    query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / np.maximum(norms, 1e-9)
    sims = (normed @ query_norm).flatten()  # (N,)

    k = min(top_k, len(names))
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    results = []
    for rank, j in enumerate(top_idx, 1):
        results.append({
            "rank": rank,
            "name": names[j],
            "path": paths[j],
            "similarity": float(sims[j]),
            "distance": float(1.0 - sims[j]),
        })
    return results


# ── Display ───────────────────────────────────────────────────────────────────

def print_recommendations(
    query_name: str,
    results: list[dict],
    encoder: str,
    dim: int,
) -> None:
    """Pretty-print the recommendation table."""
    sep = "═" * 64
    thin = "─" * 64

    print(f"\n{sep}")
    print(f"  aMOUR — Recommendations for: {query_name}")
    print(f"  Encoder: {encoder}  |  Embedding dim: {dim}")
    print(sep)

    for r in results:
        bar_len = max(0, round(r["similarity"] * 30))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(
            f"\n  {r['rank']:>2}.  {r['name']}"
            f"\n       {bar}  sim={r['similarity']:.4f}  dist={r['distance']:.4f}"
            f"\n       📁 {r['path']}"
        )

    print(f"\n{thin}")
    print(f"  {len(results)} recommendations  |  ranked by cosine similarity")
    print(f"  (computed in the original {dim}-D latent space, not UMAP)\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="aMOUR – get music recommendations for a track",
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Track name (or substring) to get recommendations for",
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        help="Path to a NEW audio file — encode on-the-fly and recommend",
    )
    parser.add_argument(
        "--encoder", type=str, default="mert",
        help="Encoder backend for --file encoding (mert | clap | music2vec)",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to embeddings .npz (default: embeddings/<encoder>.npz)",
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Number of recommendations to return",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_tracks",
        help="List all track names in the index and exit",
    )
    args = parser.parse_args()

    # Default input path
    if args.input is None:
        args.input = Path(f"embeddings/{args.encoder.lower()}.npz")

    if not args.input.exists():
        print(f"[ERROR] Embeddings not found: {args.input}")
        print("Run  python src/encode.py  first.")
        return

    matrix, names, paths, encoder = load_embeddings(args.input)
    dim = matrix.shape[1]
    print(f"Loaded {len(names)} tracks  |  encoder: {encoder}  |  dim: {dim}")

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list_tracks:
        print(f"\nTracks in {args.input}:")
        for i, (n, p) in enumerate(zip(names, paths)):
            print(f"  {i:>4}. {n:<40s}  {p}")
        return

    # ── File mode: encode a new track on-the-fly ──────────────────────────────
    if args.file is not None:
        if not args.file.exists():
            print(f"[ERROR] File not found: {args.file}")
            return

        import torch
        import librosa
        from encoders import get_encoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = get_encoder(args.encoder, device=device)

        print(f"\nEncoding {args.file.name} with {enc.name} …")
        audio, _ = librosa.load(str(args.file), sr=enc.sample_rate, mono=True)

        # Chunk
        chunk_len = enc.chunk_duration_s * enc.sample_rate
        overlap = 5 * enc.sample_rate
        hop = chunk_len - overlap
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_len, len(audio))
            chunk = audio[start:end]
            if len(chunk) < enc.sample_rate:
                break
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            chunks.append(chunk)
            if end == len(audio):
                break
            start += hop

        if not chunks:
            print("[ERROR] Audio too short to encode.")
            return

        query_vec = enc.encode_chunks(chunks, batch_size=4)
        results = recommend_for_vector(query_vec, matrix, names, paths, args.top_k)
        print_recommendations(args.file.name, results, encoder, dim)
        return

    # ── Query mode: find track in index ───────────────────────────────────────
    if args.query is None:
        parser.print_help()
        print("\nExamples:")
        print('  python src/recommend.py "song name"')
        print('  python src/recommend.py --file new_song.mp3')
        print('  python src/recommend.py --list')
        return

    idx = resolve_query(args.query, names)
    if idx is None:
        print(f"\n[ERROR] No track matching '{args.query}'")
        print(f"Use  --list  to see all {len(names)} track names.")

        # Show close suggestions
        from difflib import get_close_matches
        suggestions = get_close_matches(
            args.query.lower(),
            [n.lower() for n in names],
            n=5, cutoff=0.3,
        )
        if suggestions:
            print("\nDid you mean:")
            for s in suggestions:
                # Map back to original case
                orig = names[[n.lower() for n in names].index(s)]
                print(f"  • {orig}")
        return

    query_name = names[idx]
    print(f"Matched: '{query_name}' (index {idx})")

    results = recommend_for_index(idx, matrix, names, paths, args.top_k)
    print_recommendations(query_name, results, encoder, dim)


if __name__ == "__main__":
    main()
