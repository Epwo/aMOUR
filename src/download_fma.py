"""
aMOUR – download_fma.py
Downloads the FMA-small dataset (8,000 tracks × 30 s, ~7.2 GB).

FMA (Free Music Archive) contains CC-licensed tracks organised by genre,
making it ideal for evaluating whether latent-space clusters match musical
similarity.

Usage:
    python src/download_fma.py                          # → data/fma_small/
    python src/download_fma.py --output data/fma_small  # custom output dir
    python src/download_fma.py --skip_metadata           # audio only

Structure after download:
    data/fma_small/
    ├── 000/        ← tracks 000002.mp3 … 000999.mp3
    ├── 001/
    ├── …
    └── 155/

    data/fma_metadata/
    ├── tracks.csv   ← genre labels, artist, title, …
    ├── genres.csv
    └── …
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


FMA_AUDIO_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"


def _download(url: str, dest: Path) -> None:
    """Download with a simple progress indicator."""
    print(f"  Downloading {url}")
    print(f"  → {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:,.0f} / {total_mb:,.0f} MB ({pct}%)", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=_progress)
    print()


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Done ({sum(1 for _ in extract_to.rglob('*') if _.is_file())} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="aMOUR – download FMA-small dataset")
    parser.add_argument(
        "--output", type=Path, default=Path("data/fma_small"),
        help="Directory where audio files will be placed",
    )
    parser.add_argument(
        "--metadata_dir", type=Path, default=Path("data/fma_metadata"),
        help="Directory for FMA metadata CSVs (genre labels, etc.)",
    )
    parser.add_argument(
        "--skip_metadata", action="store_true",
        help="Only download audio, skip metadata CSVs",
    )
    parser.add_argument(
        "--keep_zip", action="store_true",
        help="Keep the .zip files after extraction",
    )
    args = parser.parse_args()

    tmp_dir = Path("data/.tmp_download")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Audio ─────────────────────────────────────────────────────────────────
    if args.output.exists() and any(args.output.rglob("*.mp3")):
        n = sum(1 for _ in args.output.rglob("*.mp3"))
        print(f"Audio already present: {n} mp3 files in {args.output}")
        print("Use --output to a different path or delete the folder to re-download.\n")
    else:
        zip_path = tmp_dir / "fma_small.zip"
        if not zip_path.exists():
            print("── Downloading FMA-small audio (~7.2 GB) ──")
            _download(FMA_AUDIO_URL, zip_path)
        else:
            print(f"Using cached {zip_path}")

        print("\n── Extracting audio ──")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _extract_zip(zip_path, args.output.parent)

        if not args.keep_zip:
            zip_path.unlink()
            print(f"  Removed {zip_path}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    if not args.skip_metadata:
        if args.metadata_dir.exists() and (args.metadata_dir / "tracks.csv").exists():
            print(f"Metadata already present in {args.metadata_dir}\n")
        else:
            zip_path = tmp_dir / "fma_metadata.zip"
            if not zip_path.exists():
                print("\n── Downloading FMA metadata (~342 MB) ──")
                _download(FMA_META_URL, zip_path)
            else:
                print(f"Using cached {zip_path}")

            print("\n── Extracting metadata ──")
            args.metadata_dir.parent.mkdir(parents=True, exist_ok=True)
            _extract_zip(zip_path, args.metadata_dir.parent)

            if not args.keep_zip:
                zip_path.unlink()
                print(f"  Removed {zip_path}")

    # Cleanup tmp dir if empty
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print("\n── Summary ──")
    if args.output.exists():
        n_mp3 = sum(1 for _ in args.output.rglob("*.mp3"))
        print(f"  Audio:    {n_mp3} tracks in {args.output}")
    if not args.skip_metadata and args.metadata_dir.exists():
        print(f"  Metadata: {args.metadata_dir}")
    print(f"\nNext step: python src/encode.py --audio_dir {args.output}")


if __name__ == "__main__":
    main()
