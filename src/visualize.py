"""
aMOUR – visualize.py
Loads precomputed embeddings, reduces to 2D/3D with UMAP,
and renders an interactive Plotly scatter plot.

Neighbour distances are computed in the ORIGINAL high-dimensional space
(before UMAP), so the top-K ranking is always geometrically accurate.

Usage:
    python src/visualize.py                          # 3D + top-5, opens browser
    python src/visualize.py --dim 2                  # 2D
    python src/visualize.py --top_k 10               # show top-10 in hover
    python src/visualize.py --edges                  # draw lines to top-3 neighbours
    python src/visualize.py --save plot.html         # save instead of opening browser
    python src/visualize.py --input embeddings/embeddings.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap

from similarity import top_k_neighbors, neighbors_to_hover_str


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_label(track_path: str) -> str:
    """
    If audio files are in sub-folders (data/rock/song.mp3 → "rock"),
    use the immediate parent as a genre/group label.
    """
    parts = Path(track_path).parts
    return parts[-2] if len(parts) >= 2 else Path(track_path).stem


def _edge_traces_2d(
    df: pd.DataFrame,
    neighbours: dict[str, list[tuple[str, float]]],
    edge_k: int,
) -> list[go.Scatter]:
    """Return one Scatter trace per edge (2-D). Grouped into a single trace for speed."""
    xs, ys = [], []
    name_to_row = {row.name_col: row for row in df.itertuples()}

    for track, tops in neighbours.items():
        if track not in name_to_row:
            continue
        src = name_to_row[track]
        for neighbour, _ in tops[:edge_k]:
            if neighbour not in name_to_row:
                continue
            dst = name_to_row[neighbour]
            xs += [src.x, dst.x, None]
            ys += [src.y, dst.y, None]

    return [go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.12)", width=1),
        hoverinfo="skip",
        showlegend=False,
        name="similarity edges",
    )]


def _edge_traces_3d(
    df: pd.DataFrame,
    neighbours: dict[str, list[tuple[str, float]]],
    edge_k: int,
) -> list[go.Scatter3d]:
    xs, ys, zs = [], [], []
    name_to_row = {row.name_col: row for row in df.itertuples()}

    for track, tops in neighbours.items():
        if track not in name_to_row:
            continue
        src = name_to_row[track]
        for neighbour, _ in tops[:edge_k]:
            if neighbour not in name_to_row:
                continue
            dst = name_to_row[neighbour]
            xs += [src.x, dst.x, None]
            ys += [src.y, dst.y, None]
            zs += [src.z, dst.z, None]

    return [go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.10)", width=1),
        hoverinfo="skip",
        showlegend=False,
        name="similarity edges",
    )]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="aMOUR – latent space visualiser")
    parser.add_argument("--input", type=Path, default=Path("embeddings/embeddings.npz"),
                        help="Path to the .npz embeddings archive produced by encode.py")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3,
                        help="Dimensionality of the UMAP projection (2 or 3)")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors — larger = more global structure")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist — smaller = tighter clusters")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Distance metric for UMAP (cosine | euclidean | …)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible UMAP layouts")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of nearest neighbours shown in hover tooltip "
                             "(computed in the original embedding space)")
    parser.add_argument("--edges", action="store_true",
                        help="Draw edge lines between each point and its top-3 "
                             "neighbours (in original space, visualised in UMAP coords)")
    parser.add_argument("--edge_k", type=int, default=3,
                        help="How many neighbour edges to draw per point (requires --edges)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save the plot to this HTML file instead of opening browser")
    args = parser.parse_args()

    # ── Load embeddings ───────────────────────────────────────────────────────
    if not args.input.exists():
        print(f"Embeddings file not found: {args.input}")
        print("Run  python src/encode.py --audio_dir data/  first.")
        return

    archive = np.load(args.input, allow_pickle=True)
    embedding_keys = [k for k in archive.files if not k.startswith("_meta_")]

    if not embedding_keys:
        print("No embeddings found in the archive.")
        return

    matrix = np.stack([archive[k] for k in embedding_keys])   # (N, H)

    names = (
        [str(n) for n in archive["_meta_names"]]
        if "_meta_names" in archive.files
        else embedding_keys
    )
    paths = (
        [str(p) for p in archive["_meta_paths"]]
        if "_meta_paths" in archive.files
        else embedding_keys
    )

    labels = [infer_label(p) for p in paths]

    print(f"Loaded {len(names)} tracks  |  embedding dim: {matrix.shape[1]}")
    print(f"Unique labels: {sorted(set(labels))}\n")

    # ── Top-K neighbours in original latent space ─────────────────────────────
    print(f"Computing top-{args.top_k} neighbours (cosine similarity, original {matrix.shape[1]}-D space) …")
    neighbours = top_k_neighbors(matrix, names, k=args.top_k)

    # Format as hover strings
    hover_neighbours = [
        neighbors_to_hover_str(neighbours[n]) for n in names
    ]
    print("Done.\n")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print(
        f"Running UMAP {matrix.shape[1]}D → {args.dim}D  "
        f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric}) …"
    )
    reducer = umap.UMAP(
        n_components=args.dim,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        verbose=True,
    )
    projected = reducer.fit_transform(matrix)   # (N, dim)
    print("UMAP done.\n")

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "name_col": names,
        "label": labels,
        "path": paths,
        "neighbours": hover_neighbours,
    })
    if args.dim == 3:
        df["x"], df["y"], df["z"] = projected[:, 0], projected[:, 1], projected[:, 2]
    else:
        df["x"], df["y"] = projected[:, 0], projected[:, 1]

    # ── Plotly ────────────────────────────────────────────────────────────────
    # Build a custom hover template so the neighbour block renders nicely
    hover_template_2d = (
        "<b>%{customdata[0]}</b><br>"
        "Folder: %{customdata[1]}<br>"
        "Path: %{customdata[2]}<br><br>"
        "%{customdata[3]}"
        "<extra></extra>"
    )
    hover_template_3d = hover_template_2d   # same structure

    custom_cols = ["name_col", "label", "path", "neighbours"]

    common_kwargs = dict(
        data_frame=df,
        color="label",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark",
        custom_data=custom_cols,
    )

    if args.dim == 3:
        fig = px.scatter_3d(
            **common_kwargs,
            x="x", y="y", z="z",
            title="aMOUR — Music Latent Space (MERT + UMAP 3D)",
        )
        fig.update_traces(
            marker=dict(size=5, opacity=0.85, line=dict(width=0)),
            hovertemplate=hover_template_3d,
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="UMAP-1",
                yaxis_title="UMAP-2",
                zaxis_title="UMAP-3",
            ),
        )
        if args.edges:
            for trace in _edge_traces_3d(df, neighbours, args.edge_k):
                fig.add_trace(trace)

    else:
        fig = px.scatter(
            **common_kwargs,
            x="x", y="y",
            title="aMOUR — Music Latent Space (MERT + UMAP 2D)",
        )
        fig.update_traces(
            marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="white")),
            hovertemplate=hover_template_2d,
        )
        fig.update_layout(xaxis_title="UMAP-1", yaxis_title="UMAP-2")
        if args.edges:
            for trace in _edge_traces_2d(df, neighbours, args.edge_k):
                fig.add_trace(trace)

    fig.update_layout(
        legend_title_text="Folder / Genre",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=0, r=0, t=50, b=0),
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            font_size=12,
            font_family="monospace",
            namelength=0,
        ),
    )

    note_parts = [f"{len(df)} tracks"]
    if args.edges:
        note_parts.append(f"edges = top-{args.edge_k} neighbours (original space)")
    fig.add_annotation(
        text="  ·  ".join(note_parts),
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        showarrow=False,
        font=dict(size=11, color="#aaaaaa"),
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(args.save))
        print(f"Plot saved → {args.save}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
