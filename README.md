# aMOUR — a Music Oriented User Recommendation (POC)

Content-based music recommendation: encode tracks with a **music** foundation
model, then recommend by cosine similarity in its latent space.

## Setup

```bash
pip install uv
uv sync          # CUDA 12.4 torch wheels; edit [tool.uv.index] in pyproject.toml for another CUDA / CPU
```

## Encoders

All are trained on music (not generic audio), except CLAP which stays as a baseline.

| key     | model                         | year | dim  | input | text search | notes |
|---------|-------------------------------|------|------|-------|-------------|-------|
| `muq`   | MuQ-large-msd (Tencent)       | 2025 | 1024 | 24 kHz | –          | **default**; SSL, beats MERT on nearly all MARBLE tasks |
| `mulan` | MuQ-MuLan-large (Tencent)     | 2025 | 512  | 24 kHz | ✓          | MuQ + contrastive text alignment; strongest on perceptual similarity |
| `omar`  | OMAR-RQ multicodebook (MTG-UPF) | 2025 | 1024 | 16 kHz | –        | SSL on 330k h of music |
| `mert`  | MERT-v1-330M (m-a-p)          | 2023 | 1024 | 24 kHz | –          | previous default |
| `clap`  | LAION CLAP larger_clap_music  | 2023 | 512  | 48 kHz | ✓          | general audio–text baseline |

For the SSL models (`muq`, `omar`, `mert`) the track vector averages every
hidden layer over time by default (genre/mood info lives in the middle
layers, not only the last). `--layers 4-8` pools a subset instead.

All model weights are **non-commercial** licences (CC-BY-NC / CC-BY-NC-SA).

## Workflow

```bash
# 1. Get audio — your Deezer likes (via YouTube), or the FMA-small benchmark set
uv run python src/fetch_deezer.py <deezer_user_id> --download   # → data/deezer/
uv run python src/download_fma.py                               # → data/fma_small/ + genre metadata

# 2. Encode (resume-safe) → embeddings/<encoder>_<dataset>.npz
uv run python src/encode.py --audio_dir data/deezer --encoder muq
uv run python src/encode.py --audio_dir data/fma_small --encoder mulan --limit 1000

# 3. Recommend
uv run python src/recommend.py "daft punk" --encoder muq
uv run python src/recommend.py --file some_new_song.mp3 --encoder muq
uv run python src/recommend.py --text "dreamy shoegaze with female vocals" --encoder mulan

# 4. Look at the latent space
uv run python src/visualize.py --encoder muq --edges

# 5. Benchmark encoders against each other (genre P@10 on shared tracks)
uv run python src/compare_encoders.py embeddings/*_fma_small.npz --plot
```

Labels (plot colours, benchmark ground truth) come from FMA's `tracks.csv`
genres when available, otherwise from the first sub-folder under `--audio_dir`.

## Layout

```
src/
  common.py            audio loading/chunking, archive I/O, labels, name lookup
  similarity.py        cosine top-k, P@k, cohesion
  encoders/            one file per backend, all implementing MusicEncoder
  encode.py            audio dir → .npz
  recommend.py         track / file / text → nearest tracks
  visualize.py         UMAP + Plotly
  compare_encoders.py  benchmark + side-by-side UMAP
  fetch_deezer.py      Deezer likes → YouTube downloads
  download_fma.py      FMA-small + metadata
```

## Ideas next

- Stem-aware similarity: separate with Demucs, embed each stem with MuQ-MuLan,
  learn per-stem weights (reported 72% → 90% agreement with human judgements, arXiv:2601.19109)
- FAISS index + FastAPI endpoint
- User taste vector = mean of liked tracks; re-rank with feedback
