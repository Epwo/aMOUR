# aMOUR — a Music Oriented User Recommendation (POC)

Content-based music recommendation via MERT latent-space embeddings.

## Setup

```bash
# install uv if needed
pip install uv

# create venv & install all deps (CUDA 12.1 torch wheels)
uv sync
```

> If your CUDA version differs, edit `[tool.uv]` `index-url` in `pyproject.toml`
> (cu118 for CUDA 11.8, cpu for no GPU).

## Workflow

### 1 — Drop your audio files into `data/`

Organise by sub-folder if you want automatic colour-coding in the plot:

```
data/
  rock/    song1.mp3  song2.flac
  jazz/    track1.wav
  ambient/ pad.ogg
```

Any flat structure works too. Supported: `.mp3 .wav .flac .ogg .m4a .aac .opus`

### 2 — Encode

```bash
uv run python src/encode.py
# options:
#   --audio_dir  data/          root of your audio files (default: data/)
#   --output     embeddings/embeddings.npz
#   --batch_size 4              chunks per GPU batch (lower if OOM, default: 4)
#   --chunk_s    30             seconds per chunk
#   --force                     re-encode already-done tracks
```

First run downloads MERT-v1-330M (~1.3 GB) from HuggingFace and caches it.
Subsequent runs skip already-encoded tracks (incremental / resume-safe).

### 3 — Visualise

```bash
uv run python src/visualize.py           # 3D interactive, opens browser
uv run python src/visualize.py --dim 2   # 2D
uv run python src/visualize.py --save plots/latent.html  # save to file
```

Hover over any point to see track name, folder label, and file path.

## Architecture

```
audio files  →  librosa (resample to 24 kHz, chunk 30 s)
             →  MERT-v1-330M (GPU batch inference)
             →  mean-pool hidden states  →  1024-dim vector per track
             →  .npz archive

.npz  →  UMAP (cosine, 3D)  →  Plotly scatter3d (dark theme)
```

## Next steps (beyond POC)

- FAISS index for real-time k-NN lookup
- FastAPI recommendation endpoint
- Optional CLAP layer for text-to-music search ("find me something sad and jazzy")
- User feedback loop / re-ranking
