"""
aMOUR – similarity.py
Similarity math in the ORIGINAL embedding space (never in UMAP space:
UMAP preserves topology, not distances).
"""

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def cosine_sim_matrix(matrix: np.ndarray) -> np.ndarray:
    """(N, H) → (N, N) pairwise cosine similarity."""
    normed = l2_normalize(matrix)
    return normed @ normed.T


def top_k(scores: np.ndarray, k: int, exclude: int | None = None) -> list[tuple[int, float]]:
    """Indices + scores of the k highest entries of a 1-D score vector, best first."""
    scores = scores.astype(np.float32, copy=True)
    if exclude is not None:
        scores[exclude] = -np.inf
    k = min(k, len(scores) - (exclude is not None))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]


def query_by_vector(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k rows of matrix closest to an arbitrary query vector."""
    return top_k(l2_normalize(matrix) @ l2_normalize(query), k)


def query_by_index(idx: int, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k neighbours of row idx (excluding itself)."""
    return top_k(l2_normalize(matrix) @ l2_normalize(matrix[idx]), k, exclude=idx)


def all_top_k(matrix: np.ndarray, k: int) -> np.ndarray:
    """(N, k) neighbour indices for every row, excluding self."""
    sim = cosine_sim_matrix(matrix)
    np.fill_diagonal(sim, -np.inf)
    k = min(k, len(sim) - 1)
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    order = np.argsort(np.take_along_axis(sim, idx, axis=1), axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1)


# ── Encoder quality metrics (need ≥ 2 distinct labels) ───────────────────────

def precision_at_k(matrix: np.ndarray, labels: list[str], k: int = 10) -> float:
    """
    Mean fraction of each track's k nearest neighbours that share its label.
    This is what a recommender actually feels like; chance level = Σ p_label².
    """
    labels_arr = np.asarray(labels)
    nn = all_top_k(matrix, k)
    return float((labels_arr[nn] == labels_arr[:, None]).mean())


def chance_precision(labels: list[str]) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float((p ** 2).sum())


def cohesion(matrix: np.ndarray, labels: list[str]) -> float:
    """Mean intra-label cosine sim minus mean inter-label cosine sim."""
    sim = cosine_sim_matrix(matrix)
    labels_arr = np.asarray(labels)
    intra, inter = [], []
    for lbl in np.unique(labels_arr):
        mask = labels_arr == lbl
        n = mask.sum()
        if n < 2 or n == len(labels_arr):
            continue
        block = sim[np.ix_(mask, mask)]
        intra.append((block.sum() - np.trace(block)) / (n * (n - 1)))
        inter.append(sim[np.ix_(mask, ~mask)].mean())
    return float(np.mean(intra) - np.mean(inter)) if intra else 0.0
