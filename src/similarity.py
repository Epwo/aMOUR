"""
aMOUR – similarity.py
Computes pairwise cosine similarity in the ORIGINAL high-dimensional embedding
space (before any UMAP reduction), which gives the true geometric proximity.

Why not use UMAP distances?
  UMAP optimises for topology preservation, not distance preservation.
  Two points that appear close in 2D/3D may not actually be the most similar
  in the 1024-D latent space.  We always rank neighbours in the raw space.

Public API
----------
    cosine_sim_matrix(matrix)           → (N, N) float32 array
    top_k_neighbors(matrix, names, k)   → dict[name → [(name, score), ...]]
    neighbors_to_hover_str(neighbors)   → "<br>"-joined string for Plotly hover
    print_top_k_report(neighbors)       → console report

Usage (standalone):
    python src/similarity.py --input embeddings/embeddings.npz --top_k 5
"""

import argparse
from pathlib import Path

import numpy as np


# ── Core math ─────────────────────────────────────────────────────────────────

def cosine_sim_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute an (N, N) pairwise cosine similarity matrix in pure NumPy.

    Parameters
    ----------
    matrix : (N, H) float array — raw embeddings (need not be normalised)

    Returns
    -------
    sim : (N, N) float32 array, values in [-1, 1]
          sim[i, j] == 1  →  identical direction (most similar)
          sim[i, j] == 0  →  orthogonal
          sim[i, j] == -1 →  opposite direction
    """
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)          # (N, 1)
    normed = matrix / np.maximum(norms, 1e-9)                      # (N, H)
    return (normed @ normed.T).astype(np.float32)                  # (N, N)


def cosine_dist_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Cosine *distance* = 1 - cosine_similarity.
    Values in [0, 2]: 0 = identical, 2 = opposite.
    """
    return 1.0 - cosine_sim_matrix(matrix)


# ── Neighbour extraction ───────────────────────────────────────────────────────

def top_k_neighbors(
    matrix: np.ndarray,
    names: list[str],
    k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """
    For each track, find the k most similar tracks in the original latent space.

    Parameters
    ----------
    matrix : (N, H) float array
    names  : list of N track names (same order as rows of matrix)
    k      : how many neighbours to return

    Returns
    -------
    neighbours : dict[track_name → [(neighbour_name, cosine_sim), ...]]
                 sorted descending by similarity (best first).
    """
    n = len(names)
    k = min(k, n - 1)          # can't have more neighbours than tracks - 1

    sim = cosine_sim_matrix(matrix)         # (N, N)
    np.fill_diagonal(sim, -np.inf)          # exclude self

    result: dict[str, list[tuple[str, float]]] = {}
    for i, name in enumerate(names):
        top_idx = np.argpartition(sim[i], -k)[-k:]         # fast top-k
        top_idx = top_idx[np.argsort(sim[i, top_idx])[::-1]]  # sort desc
        result[name] = [(names[j], float(sim[i, j])) for j in top_idx]

    return result


# ── Formatting helpers ─────────────────────────────────────────────────────────

def neighbors_to_hover_str(neighbors: list[tuple[str, float]]) -> str:
    """
    Convert a track's neighbour list into a multi-line string for Plotly hover.
    Uses HTML <br> line breaks (Plotly renders these in hover tooltips).

    Example output:
        ──────────────────
        1. Bohemian Rhapsody  (0.943)
        2. Stairway to Heaven (0.921)
        3. Hotel California   (0.907)
    """
    lines = ["<b>Nearest songs</b>", "─" * 22]
    for rank, (name, score) in enumerate(neighbors, 1):
        # Truncate very long names to keep the tooltip tidy
        display = name if len(name) <= 30 else name[:27] + "…"
        bar = "█" * round(score * 10)          # mini similarity bar
        lines.append(f"{rank}. {display}<br>   {bar} {score:.3f}")
    return "<br>".join(lines)


# ── Console report ─────────────────────────────────────────────────────────────

def print_top_k_report(
    neighbours: dict[str, list[tuple[str, float]]],
    title: str = "aMOUR — Top-K Nearest Neighbours (original latent space)",
) -> None:
    """Pretty-print the full neighbour report to stdout."""
    sep = "─" * 60
    print(f"\n{title}")
    print(sep)
    for track, tops in neighbours.items():
        print(f"\n  🎵  {track}")
        for rank, (name, score) in enumerate(tops, 1):
            bar = "█" * round(score * 20)
            print(f"       {rank}. {name:<35s}  {bar}  {score:.4f}")
    print(f"\n{sep}\n")


# ── Standalone CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="aMOUR – compute & display top-K nearest songs"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("embeddings/embeddings.npz"),
        help="Path to embeddings .npz produced by encode.py",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of nearest neighbours to display per track",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Embeddings not found: {args.input}")
        print("Run  python src/encode.py --audio_dir data/  first.")
        return

    archive = np.load(args.input, allow_pickle=True)
    embedding_keys = [k for k in archive.files if not k.startswith("_meta_")]

    if not embedding_keys:
        print("[ERROR] No embeddings found in the archive.")
        return

    matrix = np.stack([archive[k] for k in embedding_keys])  # (N, H)
    names = (
        [str(n) for n in archive["_meta_names"]]
        if "_meta_names" in archive.files
        else embedding_keys
    )

    print(f"Loaded {len(names)} tracks  |  embedding dim: {matrix.shape[1]}")
    print(f"Computing pairwise cosine similarity ({len(names)}×{len(names)}) …")

    neighbours = top_k_neighbors(matrix, names, k=args.top_k)
    print_top_k_report(neighbours)


if __name__ == "__main__":
    main()
