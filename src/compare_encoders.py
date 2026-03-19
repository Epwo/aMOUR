"""
aMOUR – compare_encoders.py
Side-by-side comparison of multiple encoder latent spaces on the same dataset.

Produces a Plotly subplot figure where each panel shows the UMAP projection
from a different encoder, letting you visually compare how well each model
clusters similar music.

Usage:
    # Compare MERT vs CLAP on your deezer library
    python src/compare_encoders.py embeddings/mert_deezer.npz embeddings/clap_deezer.npz

    # Compare all four encoders
    python src/compare_encoders.py embeddings/mert_deezer.npz embeddings/clap_deezer.npz \
        embeddings/music2vec_deezer.npz embeddings/encodec_deezer.npz

    # 2D instead of 3D, save to file
    python src/compare_encoders.py embeddings/*.npz --dim 2 --save comparison.html

    # Show top-5 neighbours per encoder for a specific track
    python src/compare_encoders.py embeddings/*.npz --query "Artist - Song"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap

from similarity import top_k_neighbors, cosine_sim_matrix


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_label(track_path: str) -> str:
    parts = Path(track_path).parts
    return parts[-2] if len(parts) >= 2 else Path(track_path).stem


def load_archive(path: Path) -> tuple[np.ndarray, list[str], list[str], str]:
    """Load .npz → (matrix, names, paths, encoder_name)."""
    archive = np.load(path, allow_pickle=True)
    keys = [k for k in archive.files if not k.startswith("_meta_")]
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
        else path.stem.split("_")[0]
    )
    return matrix, names, paths, encoder


def compute_cohesion_score(matrix: np.ndarray, labels: list[str]) -> float:
    """
    Simple cluster quality metric: mean intra-class similarity minus
    mean inter-class similarity. Higher = better genre separation.
    """
    sim = cosine_sim_matrix(matrix)
    np.fill_diagonal(sim, 0)

    unique_labels = list(set(labels))
    if len(unique_labels) < 2:
        return 0.0

    label_arr = np.array(labels)
    intra_sims, inter_sims = [], []

    for lbl in unique_labels:
        mask = label_arr == lbl
        n = mask.sum()
        if n < 2:
            continue
        intra = sim[np.ix_(mask, mask)]
        intra_sims.append(intra.sum() / (n * (n - 1)))

        inter = sim[np.ix_(mask, ~mask)]
        inter_sims.append(inter.mean())

    if not intra_sims:
        return 0.0
    return float(np.mean(intra_sims) - np.mean(inter_sims))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="aMOUR – compare encoder latent spaces",
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Two or more .npz embedding files to compare",
    )
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2,
                        help="UMAP projection dimensionality (default: 2 for comparison)")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=None,
                        help="Save to HTML instead of opening browser")
    parser.add_argument("--query", type=str, default=None,
                        help="Show top-5 neighbours for this track across all encoders")
    args = parser.parse_args()

    n_encoders = len(args.inputs)
    if n_encoders < 2:
        print("[ERROR] Provide at least 2 .npz files to compare.")
        return

    # ── Load all archives ─────────────────────────────────────────────────────
    archives = []
    for path in args.inputs:
        if not path.exists():
            print(f"[WARN] Skipping missing file: {path}")
            continue
        matrix, names, paths, encoder = load_archive(path)
        archives.append((matrix, names, paths, encoder, path))
        print(f"  Loaded {path.name}: {len(names)} tracks, dim={matrix.shape[1]}, encoder={encoder}")

    if len(archives) < 2:
        print("[ERROR] Need at least 2 valid archives.")
        return

    # ── Query mode: compare neighbours ────────────────────────────────────────
    if args.query:
        print(f"\n{'═' * 70}")
        print(f"  Comparing recommendations for: {args.query}")
        print(f"{'═' * 70}")

        for matrix, names, paths, encoder, fpath in archives:
            # Fuzzy match
            from difflib import get_close_matches
            lower_names = [n.lower() for n in names]
            ql = args.query.lower()

            idx = None
            if ql in lower_names:
                idx = lower_names.index(ql)
            else:
                for i, n in enumerate(lower_names):
                    if ql in n:
                        idx = i
                        break
            if idx is None:
                matches = get_close_matches(ql, lower_names, n=1, cutoff=0.3)
                if matches:
                    idx = lower_names.index(matches[0])

            if idx is None:
                print(f"\n  [{encoder}] Track not found in {fpath.name}")
                continue

            neighbours = top_k_neighbors(matrix, names, k=5)
            track_name = names[idx]

            print(f"\n  [{encoder}] (dim={matrix.shape[1]})")
            for rank, (name, score) in enumerate(neighbours[track_name], 1):
                bar = "█" * round(score * 20)
                print(f"    {rank}. {name:<35s}  {bar}  {score:.4f}")

        print(f"\n{'─' * 70}\n")
        return

    # ── UMAP + subplots ───────────────────────────────────────────────────────
    n = len(archives)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    if args.dim == 3:
        specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    else:
        specs = [[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]

    subtitles = []
    for matrix, names, paths, encoder, fpath in archives:
        labels = [infer_label(p) for p in paths]
        score = compute_cohesion_score(matrix, labels)
        subtitles.append(f"{encoder} (dim={matrix.shape[1]}, cohesion={score:.3f})")

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subtitles,
        specs=specs,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    colors = [
        "#E45756", "#4C78A8", "#54A24B", "#F58518",
        "#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6",
    ]

    for panel_idx, (matrix, names, paths, encoder, fpath) in enumerate(archives):
        row = panel_idx // cols + 1
        col = panel_idx % cols + 1
        labels = [infer_label(p) for p in paths]
        unique_labels = sorted(set(labels))

        print(f"\nRunning UMAP for {encoder} ({matrix.shape[1]}D → {args.dim}D) …")
        reducer = umap.UMAP(
            n_components=args.dim,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric="cosine",
            random_state=args.seed,
            verbose=False,
        )
        projected = reducer.fit_transform(matrix)

        label_arr = np.array(labels)
        for lbl_idx, lbl in enumerate(unique_labels):
            mask = label_arr == lbl
            color = colors[lbl_idx % len(colors)]

            if args.dim == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=projected[mask, 0],
                        y=projected[mask, 1],
                        z=projected[mask, 2],
                        mode="markers",
                        marker=dict(size=3, opacity=0.8, color=color),
                        name=lbl,
                        text=[names[i] for i in np.where(mask)[0]],
                        hovertemplate="<b>%{text}</b><br>%{marker.color}<extra></extra>",
                        showlegend=(panel_idx == 0),
                    ),
                    row=row, col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=projected[mask, 0],
                        y=projected[mask, 1],
                        mode="markers",
                        marker=dict(size=6, opacity=0.8, color=color),
                        name=lbl,
                        text=[names[i] for i in np.where(mask)[0]],
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        showlegend=(panel_idx == 0),
                    ),
                    row=row, col=col,
                )

    fig.update_layout(
        title="aMOUR — Encoder Comparison",
        template="plotly_dark",
        font=dict(family="Inter, sans-serif", size=12),
        height=400 * rows,
        width=450 * cols,
        legend_title_text="Folder / Genre",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(args.save))
        print(f"\nPlot saved → {args.save}")
    else:
        fig.show()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  {'Encoder':<20s}  {'Dim':>5}  {'Tracks':>6}  {'Cohesion':>9}  File")
    print(f"{'─' * 65}")
    for matrix, names, paths, encoder, fpath in archives:
        labels = [infer_label(p) for p in paths]
        score = compute_cohesion_score(matrix, labels)
        print(f"  {encoder:<20s}  {matrix.shape[1]:>5}  {len(names):>6}  {score:>+9.4f}  {fpath.name}")
    print(f"{'═' * 65}")
    print("  Cohesion = mean(intra-class sim) - mean(inter-class sim)")
    print("  Higher is better (tighter clusters, more separation)\n")


if __name__ == "__main__":
    main()
