"""
aMOUR – visualize.py
UMAP projection of an embeddings archive as an interactive Plotly scatter.
Hover shows each track's nearest neighbours, computed in the ORIGINAL
embedding space (UMAP distances are not meaningful).

Usage:
    python src/visualize.py --encoder muq                 # 3D, opens browser
    python src/visualize.py --input embeddings/muq_deezer.npz --dim 2
    python src/visualize.py --encoder muq --edges         # lines to top-3 neighbours
    python src/visualize.py --encoder muq --save plots/muq.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap

from common import load_archive, resolve_archive_path
from similarity import all_top_k, cosine_sim_matrix


def hover_block(names: list[str], neighbours: list[int], sims: np.ndarray) -> str:
    lines = ["<b>Nearest songs</b>", "─" * 22]
    for rank, j in enumerate(neighbours, 1):
        name = names[j] if len(names[j]) <= 30 else names[j][:27] + "…"
        lines.append(f"{rank}. {name}<br>   {'█' * round(sims[j] * 10)} {sims[j]:.3f}")
    return "<br>".join(lines)


def edge_trace(coords: np.ndarray, nn: np.ndarray, edge_k: int):
    segs = [[] for _ in range(coords.shape[1])]
    for i, row in enumerate(nn[:, :edge_k]):
        for j in row:
            for d in range(coords.shape[1]):
                segs[d] += [coords[i, d], coords[j, d], None]
    style = dict(mode="lines", hoverinfo="skip", showlegend=False,
                 line=dict(color="rgba(255,255,255,0.12)", width=1))
    if coords.shape[1] == 3:
        return go.Scatter3d(x=segs[0], y=segs[1], z=segs[2], **style)
    return go.Scatter(x=segs[0], y=segs[1], **style)


def main() -> None:
    parser = argparse.ArgumentParser(description="aMOUR – latent space visualiser")
    parser.add_argument("--input", type=Path, help="embeddings archive (.npz)")
    parser.add_argument("--encoder", help="picks embeddings/<encoder>_*.npz when --input is omitted")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3)
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP: larger = more global")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP: smaller = tighter clusters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=5, help="neighbours listed in the hover")
    parser.add_argument("--edges", action="store_true", help="draw lines to nearest neighbours")
    parser.add_argument("--edge_k", type=int, default=3)
    parser.add_argument("--save", type=Path, help="write HTML instead of opening a browser")
    args = parser.parse_args()

    archive = load_archive(resolve_archive_path(args.input, args.encoder))
    print(f"Loaded {len(archive)} tracks  |  {archive.encoder}  |  {archive.dim}-D")

    nn = all_top_k(archive.matrix, max(args.top_k, args.edge_k))
    sim = cosine_sim_matrix(archive.matrix)
    hovers = [hover_block(archive.names, nn[i, : args.top_k], sim[i]) for i in range(len(archive))]

    print(f"Running UMAP {archive.dim}D → {args.dim}D …")
    coords = umap.UMAP(
        n_components=args.dim, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        metric="cosine", random_state=args.seed,
    ).fit_transform(archive.matrix)

    df = pd.DataFrame({"name": archive.names, "label": archive.labels,
                       "path": archive.paths, "neighbours": hovers})
    axes = ["x", "y", "z"][: args.dim]
    for d, ax in enumerate(axes):
        df[ax] = coords[:, d]

    plot = px.scatter_3d if args.dim == 3 else px.scatter
    fig = plot(
        df, **dict(zip(axes, axes)), color="label",
        color_discrete_sequence=px.colors.qualitative.Bold, template="plotly_dark",
        custom_data=["name", "label", "path", "neighbours"],
        title=f"aMOUR — {archive.encoder} latent space (UMAP {args.dim}D)",
    )
    fig.update_traces(
        marker=dict(size=5 if args.dim == 3 else 8, opacity=0.85),
        hovertemplate="<b>%{customdata[0]}</b><br>Label: %{customdata[1]}<br>"
                      "Path: %{customdata[2]}<br><br>%{customdata[3]}<extra></extra>",
    )
    if args.edges:
        fig.add_trace(edge_trace(coords, nn, args.edge_k))
    fig.update_layout(
        legend_title_text="Label", margin=dict(l=0, r=0, t=50, b=0),
        hoverlabel=dict(bgcolor="#1a1a2e", font_family="monospace", namelength=0),
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(args.save))
        print(f"Plot saved → {args.save}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
