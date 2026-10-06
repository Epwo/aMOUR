"""
aMOUR – common.py
Shared helpers: audio loading/chunking, embedding archive I/O, labels,
track-name lookup. Everything that used to be copy-pasted across scripts.
"""

import sys
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import librosa
import numpy as np
import torch

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
EMBEDDINGS_DIR = Path("embeddings")
FMA_TRACKS_CSV = Path("data/fma_metadata/tracks.csv")

# Windows consoles default to cp1252, which can't print the box/arrow glyphs we use.
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding="utf-8", errors="replace")


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(verbose: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
        else:
            print("No GPU found — running on CPU (this will be slow)")
    return device


# ── Audio ─────────────────────────────────────────────────────────────────────

def scan_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(
        p for p in audio_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_audio(path: Path, target_sr: int) -> np.ndarray | None:
    """Load and resample audio to mono float32 @ target_sr. None on failure."""
    try:
        audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception as exc:
        print(f"  [WARN] Could not load {path.name}: {exc}")
        return None


def chunk_audio(
    audio: np.ndarray, sr: int, chunk_s: float, overlap_s: float = 5.0,
) -> list[np.ndarray]:
    """
    Split a waveform into overlapping windows of chunk_s seconds.

    Nothing is zero-padded (padding silence dilutes the mean-pooled embedding):
    a remaining tail gets its own window aligned to the end of the track.
    Tracks shorter than one window yield a single (shorter) chunk.
    """
    chunk_len = int(chunk_s * sr)
    hop_len = max(1, int((chunk_s - overlap_s) * sr))

    if len(audio) < sr:  # < 1 s: not worth encoding
        return []
    if len(audio) <= chunk_len:
        return [audio]

    starts = list(range(0, len(audio) - chunk_len + 1, hop_len))
    # Cover the tail only if it's substantial, else the last window would
    # nearly duplicate the previous one and over-weight the outro.
    if len(audio) - (starts[-1] + chunk_len) >= hop_len // 2:
        starts.append(len(audio) - chunk_len)
    return [audio[s : s + chunk_len] for s in starts]


# ── Labels ────────────────────────────────────────────────────────────────────

def _fma_genres(tracks_csv: Path) -> dict[int, str]:
    import pandas as pd

    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    genres = tracks[("track", "genre_top")].dropna()
    return {int(tid): str(g) for tid, g in genres.items()}


def infer_labels(paths: list[Path], audio_dir: Path) -> list[str]:
    """
    One label per track, used for colouring plots and scoring encoders.

    - FMA (numeric file names + data/fma_metadata/tracks.csv present):
      the real top-level genre from the metadata.
    - Otherwise: the immediate sub-folder under audio_dir
      (data/rock/song.mp3 → "rock"), or the dataset folder name if flat.
    """
    fma = _fma_genres(FMA_TRACKS_CSV) if FMA_TRACKS_CSV.exists() else {}
    labels = []
    for p in paths:
        if fma and p.stem.isdigit() and int(p.stem) in fma:
            labels.append(fma[int(p.stem)])
            continue
        rel = p.relative_to(audio_dir)
        labels.append(rel.parts[0] if len(rel.parts) > 1 else audio_dir.name)
    return labels


# ── Embedding archives ────────────────────────────────────────────────────────

@dataclass
class Archive:
    matrix: np.ndarray  # (N, D) float32
    names: list[str]
    paths: list[str]
    labels: list[str]
    encoder: str
    path: Path
    legacy: bool = False  # written by the pre-cleanup encode.py

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def __len__(self) -> int:
        return len(self.names)


def default_archive_path(encoder: str, audio_dir: Path) -> Path:
    return EMBEDDINGS_DIR / f"{encoder.lower()}_{audio_dir.name}.npz"


def save_archive(path: Path, archive: Archive) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=archive.matrix.astype(np.float32),
        names=np.array(archive.names),
        paths=np.array(archive.paths),
        labels=np.array(archive.labels),
        encoder=np.array(archive.encoder),
    )


def load_archive(path: Path) -> Archive:
    """Load an archive written by encode.py (also reads the pre-cleanup format)."""
    npz = np.load(path, allow_pickle=False)

    if "embeddings" in npz.files:
        names = [str(n) for n in npz["names"]]
        paths = [str(p) for p in npz["paths"]]
        return Archive(
            matrix=npz["embeddings"].astype(np.float32),
            names=names,
            paths=paths,
            labels=[str(l) for l in npz["labels"]],
            encoder=str(npz["encoder"]),
            path=path,
        )

    # Legacy: one array per track + _meta_* arrays
    keys = [k for k in npz.files if not k.startswith("_meta_")]
    names = [str(n) for n in npz["_meta_names"]] if "_meta_names" in npz.files else keys
    paths = [Path(p).as_posix() for p in npz["_meta_paths"]] if "_meta_paths" in npz.files else keys
    labels = [Path(p).parts[-2] if len(Path(p).parts) > 1 else path.stem for p in paths]
    return Archive(
        matrix=np.stack([npz[n] for n in names]).astype(np.float32),
        names=names,
        paths=paths,
        labels=labels,
        encoder=str(npz["_meta_encoder"]) if "_meta_encoder" in npz.files else path.stem,
        path=path,
        legacy=True,
    )


def resolve_archive_path(input_path: Path | None, encoder: str | None) -> Path:
    """
    Pick the archive to use:
      --input given            → that file
      --encoder given          → the single embeddings/<encoder>_*.npz
      neither                  → the single embeddings/*.npz
    Exits with a helpful message when ambiguous or missing.
    """
    if input_path is not None:
        if not input_path.exists():
            raise SystemExit(f"[ERROR] Embeddings not found: {input_path}")
        return input_path

    pattern = f"{encoder.lower()}_*.npz" if encoder else "*.npz"
    candidates = sorted(EMBEDDINGS_DIR.glob(pattern))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"[ERROR] No archive matching {EMBEDDINGS_DIR / pattern}. "
            "Run  python src/encode.py  first."
        )
    listing = "\n".join(f"  {c}" for c in candidates)
    raise SystemExit(f"[ERROR] Several archives match, pick one with --input:\n{listing}")


# ── Track lookup ──────────────────────────────────────────────────────────────

def resolve_query(query: str, names: list[str]) -> int | None:
    """Exact → case-insensitive → substring → fuzzy. Returns an index or None."""
    if query in names:
        return names.index(query)
    lower = [n.lower() for n in names]
    q = query.lower()
    if q in lower:
        return lower.index(q)
    for i, n in enumerate(lower):
        if q in n:
            return i
    matches = get_close_matches(q, lower, n=1, cutoff=0.4)
    return lower.index(matches[0]) if matches else None


def suggest_names(query: str, names: list[str], n: int = 5) -> list[str]:
    lower = [x.lower() for x in names]
    return [names[lower.index(m)] for m in get_close_matches(query.lower(), lower, n=n, cutoff=0.3)]
