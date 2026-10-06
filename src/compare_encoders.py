"""
aMOUR – compare_encoders.py
Benchmark several encoders on the same tracks.

Scores (on the tracks present in every archive):
  P@10      fraction of each track's 10 nearest neighbours sharing its label
            (genre on FMA, sub-folder otherwise) — what a recommender "feels" like
  cohesion  mean intra-label cosine sim − mean inter-label cosine sim

Usage:
    python src/compare_encoders.py embeddings/*_fma_small.npz           # score table
    python src/compare_encoders.py embeddings/*_fma_small.npz --plot    # + UMAP panels
    python src/compare_encoders.py embeddings/*_deezer.npz --query "daft punk"
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots

from common import Archive, load_archive, resolve_query
from similarity import chance_precision, cohesion, precision_at_k, query_by_index


def align(archives: list[Archive]) -> list[Archive]:
    """Restrict every archive to the tracks they all contain, in the same order."""
    common = set.intersection(*(set(a.paths) for a in archives))
    order = sorted(common)
    out = []
    for a in archives:
        pos = {p: i for i, p in enumerate(a.paths)}
        idx = [pos[p] for p in order]
        out.append(Archive(a.matrix[idx], [a.names[i] for i in idx], order,
                           [a.labels[i] for i in idx], a.encoder, a.path))
    return out


def score_table(archives: list[Archive], k: int) -> None:
    labels = archives[0].labels
    n_labels = len(set(labels))
    print(f"\n{len(archives[0])} shared tracks, {n_labels} labels")
    if n_labels < 2:
        print("Only one label — scores need ≥ 2 (use FMA or genre sub-folders).")
        return
    print(f"\n  {'Encoder':<24s} {'Dim':>5}  {f'P@{k}':>6}  {'Cohesion':>8}  File")
    print("  " + "─" * 70)
    rows = [(a, precision_at_k(a.matrix, labels, k), cohesion(a.matrix, labels)) for a in archives]
    for a, p, c in sorted(rows, key=lambda r: -r[1]):
        print(f"  {a.encoder:<24s} {a.dim:>5}  {p:>6.3f}  {c:>+8.4f}  {a.path.name}")
    print(f"\n  chance P@{k} = {chance_precision(labels):.3f}\n")


def query_report(archives: list[Archive], query: str, k: int) -> None:
    idx = resolve_query(query, archives[0].names)
    if idx is None:
        raise SystemExit(f"[ERROR] No shared track matching '{query}'.")
    print(f"\nNeighbours of: {archives[0].names[idx]}")
    for a in archives:
        print(f"\n  [{a.encoder}]")
        for rank, (j, s) in enumerate(query_by_index(idx, a.matrix, k), 1):
            print(f"    {rank:>2}. {a.names[j]:<45s} {s:.3f}  [{a.labels[j]}]")
    print()


def plot_panels(archives: list[Archive], save: Path | None, seed: int) -> None:
    cols = min(len(archives), 3)
    rows = -(-len(archives) // cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[a.encoder for a in archives],
                        horizontal_spacing=0.04, vertical_spacing=0.08)
    labels = np.array(archives[0].labels)
    palette = px.colors.qualitative.Bold
    for n, a in enumerate(archives):
        print(f"UMAP for {a.encoder} …")
        xy = umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(a.matrix)
        for li, lbl in enumerate(sorted(set(labels))):
            m = labels == lbl
            fig.add_trace(
                go.Scatter(x=xy[m, 0], y=xy[m, 1], mode="markers", name=lbl, legendgroup=lbl,
                           marker=dict(size=5, opacity=0.8, color=palette[li % len(palette)]),
                           text=np.array(a.names)[m], hovertemplate="<b>%{text}</b><extra></extra>",
                           showlegend=(n == 0)),
                row=n // cols + 1, col=n % cols + 1,
            )
    fig.update_layout(title="aMOUR — encoder comparison", template="plotly_dark",
                      height=420 * rows, width=460 * cols, legend_title_text="Label")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save))
        print(f"Plot saved → {save}")
    else:
        fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="aMOUR – compare encoder latent spaces")
    parser.add_argument("inputs", nargs="+", type=Path, help="two or more .npz archives")
    parser.add_argument("--k", type=int, default=10, help="neighbours for P@k / --query")
    parser.add_argument("--query", help="show each encoder's neighbours for this track")
    parser.add_argument("--plot", action="store_true", help="UMAP panel per encoder")
    parser.add_argument("--save", type=Path, help="write the plot to HTML (implies --plot)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archives = [load_archive(p) for p in args.inputs if p.exists()]
    if len(archives) < 2:
        raise SystemExit("[ERROR] Need at least 2 existing archives.")
    archives = align(archives)
    if not len(archives[0]):
        raise SystemExit("[ERROR] The archives share no tracks.")

    if args.query:
        query_report(archives, args.query, args.k)
        return
    score_table(archives, args.k)
    if args.plot or args.save:
        plot_panels(archives, args.save, args.seed)


if __name__ == "__main__":
    main()
