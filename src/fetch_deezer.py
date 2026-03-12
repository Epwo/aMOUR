"""
aMOUR – fetch_deezer.py
Fetches your Deezer liked tracks list, then downloads the full audio
from YouTube via yt-dlp (much more reliable than Deezer previews).

No API key required — works with any public Deezer profile.

Usage:
    # List liked tracks
    python src/fetch_deezer.py 700513741
    python src/fetch_deezer.py https://www.deezer.com/profile/700513741

    # Search for a user by name
    python src/fetch_deezer.py --search "John Doe"

    # Download from YouTube into data/deezer/
    python src/fetch_deezer.py 700513741 --download

    # Limit to first N tracks
    python src/fetch_deezer.py 700513741 --download --limit 50

    # Export the track list as CSV
    python src/fetch_deezer.py 700513741 --csv likes.csv

    # Custom output dir + max duration (skip long mixes)
    python src/fetch_deezer.py 700513741 --download --output data/my_likes --max_duration 600
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

API_BASE = "https://api.deezer.com"
PAGE_SIZE = 100  # max per Deezer API request


# ── Deezer API helpers ───────────────────────────────────────────────────────

def _api_get(url: str) -> dict:
    """GET a Deezer API endpoint and return parsed JSON."""
    req = Request(url, headers={"User-Agent": "aMOUR/0.1"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    if "error" in data:
        raise RuntimeError(
            f"Deezer API error: {data['error'].get('message', data['error'])}"
        )
    return data


def search_users(query: str, limit: int = 10) -> list[dict]:
    """Search for Deezer users by name."""
    data = _api_get(f"{API_BASE}/search/user?q={query}&limit={limit}")
    return data.get("data", [])


def get_user_info(user_id: int) -> dict:
    """Get basic profile info for a user ID."""
    return _api_get(f"{API_BASE}/user/{user_id}")


def get_liked_tracks(user_id: int, limit: int | None = None) -> list[dict]:
    """
    Fetch all liked/favorite tracks for a user (paginated).
    Returns a list of track dicts.
    """
    tracks = []
    url = f"{API_BASE}/user/{user_id}/tracks?limit={PAGE_SIZE}"

    while url:
        data = _api_get(url)
        batch = data.get("data", [])
        if not batch:
            break
        tracks.extend(batch)

        if limit and len(tracks) >= limit:
            tracks = tracks[:limit]
            break

        url = data.get("next")
        time.sleep(0.2)

    return tracks


# ── Parse user ID from various inputs ─────────────────────────────────────────

def parse_user_id(input_str: str) -> int | None:
    """
    Extract a Deezer user ID from:
      - A plain numeric ID: "700513741"
      - A profile URL: "https://www.deezer.com/profile/700513741"
    """
    try:
        return int(input_str.strip())
    except ValueError:
        pass

    match = re.search(r"deezer\.com/(?:\w+/)?profile/(\d+)", input_str)
    if match:
        return int(match.group(1))

    return None


# ── Sanitise filenames ────────────────────────────────────────────────────────

def _safe_filename(name: str, max_len: int = 80) -> str:
    """Replace filesystem-unsafe chars and limit length."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe = safe.strip(". ")
    return safe[:max_len] if len(safe) > max_len else safe


# ── YouTube download via yt-dlp ──────────────────────────────────────────────

def _download_one_yt(
    track: dict,
    output_dir: Path,
    max_duration: int,
) -> tuple[str, bool, str]:
    """
    Search YouTube for "artist - title" and download as mp3.
    Returns (display_name, success, message).
    """
    import yt_dlp

    artist = track.get("artist", {}).get("name", "Unknown")
    title = track.get("title", "Unknown")
    display = f"{artist} - {title}"
    filename = _safe_filename(display)

    # Skip if already downloaded (check for any audio extension)
    for ext in (".mp3", ".m4a", ".opus", ".wav", ".webm"):
        if (output_dir / f"{filename}{ext}").exists():
            return display, True, "already exists"

    search_query = f"ytsearch1:{artist} - {title} audio"

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": str(output_dir / f"{filename}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration < {max_duration}"
        ) if max_duration else None,
        # Don't download playlists
        "noplaylist": True,
        # Retry on failure
        "retries": 3,
        "fragment_retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([search_query])
        return display, True, "downloaded"
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc).split("\n")[0][:80]
        return display, False, msg
    except Exception as exc:
        return display, False, str(exc)[:80]


def download_tracks(
    tracks: list[dict],
    output_dir: Path,
    max_duration: int = 600,
) -> tuple[int, int, int]:
    """
    Download tracks from YouTube sequentially.
    (yt-dlp handles its own parallelism internally; running multiple
     instances risks rate-limiting from YouTube.)

    Returns (success_count, fail_count, skip_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, fail, skip = 0, 0, 0
    total = len(tracks)

    for i, track in enumerate(tracks, 1):
        display, success, msg = _download_one_yt(track, output_dir, max_duration)

        if msg == "already exists":
            icon = "⊘"
            skip += 1
        elif success:
            icon = "✓"
            ok += 1
        else:
            icon = "✗"
            fail += 1

        print(f"  [{i:>4}/{total}] {icon}  {display}  ({msg})")

    return ok, fail, skip


# ── Display ───────────────────────────────────────────────────────────────────

def print_track_list(tracks: list[dict]) -> None:
    """Pretty-print the liked tracks."""
    sep = "─" * 70
    print(f"\n{'#':>4}  {'Artist':<25}  {'Title':<30}  {'Duration':>5}")
    print(sep)
    for i, t in enumerate(tracks, 1):
        artist = t.get("artist", {}).get("name", "?")[:25]
        title = t.get("title", "?")[:30]
        dur = t.get("duration", 0)
        mins = dur // 60
        secs = dur % 60
        print(f"{i:>4}  {artist:<25}  {title:<30}  {mins}:{secs:02d}")
    print(sep)
    print(f"  {len(tracks)} tracks\n")


def export_csv(tracks: list[dict], csv_path: Path) -> None:
    """Export the track list as CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "artist", "title", "album", "duration_s", "deezer_id", "deezer_url",
        ])
        writer.writeheader()
        for t in tracks:
            writer.writerow({
                "artist": t.get("artist", {}).get("name", ""),
                "title": t.get("title", ""),
                "album": t.get("album", {}).get("title", ""),
                "duration_s": t.get("duration", 0),
                "deezer_id": t.get("id", ""),
                "deezer_url": t.get("link", ""),
            })
    print(f"Exported {len(tracks)} tracks → {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="aMOUR – fetch Deezer liked tracks & download from YouTube",
    )
    parser.add_argument(
        "user", nargs="?", default=None,
        help="Deezer user ID or profile URL (e.g. 700513741 or "
             "https://www.deezer.com/profile/700513741)",
    )
    parser.add_argument(
        "--search", type=str, default=None,
        help="Search for a Deezer user by name instead of providing an ID",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download audio from YouTube for all liked tracks",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/deezer"),
        help="Output directory for downloaded audio (default: data/deezer/)",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, dest="csv_path",
        help="Export the track list to a CSV file",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of tracks to fetch",
    )
    parser.add_argument(
        "--max_duration", type=int, default=600,
        help="Skip YouTube results longer than this (seconds, default: 600). "
             "Prevents downloading hour-long mixes by accident.",
    )
    args = parser.parse_args()

    # ── User search mode ──────────────────────────────────────────────────────
    if args.search:
        print(f"Searching Deezer users for '{args.search}' …\n")
        users = search_users(args.search)
        if not users:
            print("No users found.")
            return
        print(f"{'ID':<15}  {'Name':<30}  {'Country'}")
        print("─" * 55)
        for u in users:
            print(f"{u['id']:<15}  {u.get('name', '?'):<30}  {u.get('country', '?')}")
        print(f"\nUse:  python src/fetch_deezer.py <ID> --download")
        return

    # ── Resolve user ID ───────────────────────────────────────────────────────
    if args.user is None:
        parser.print_help()
        print("\nExamples:")
        print("  python src/fetch_deezer.py 700513741")
        print("  python src/fetch_deezer.py https://www.deezer.com/profile/700513741")
        print("  python src/fetch_deezer.py --search 'John Doe'")
        return

    user_id = parse_user_id(args.user)
    if user_id is None:
        print(f"[ERROR] Could not parse user ID from: {args.user}")
        print("Provide a numeric ID or a deezer.com/profile/ URL.")
        return

    # ── Fetch user info ───────────────────────────────────────────────────────
    try:
        user_info = get_user_info(user_id)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    user_name = user_info.get("name", "Unknown")
    print(f"\nDeezer user: {user_name} (ID: {user_id})")

    # ── Fetch liked tracks ────────────────────────────────────────────────────
    print("Fetching liked tracks …")
    tracks = get_liked_tracks(user_id, limit=args.limit)

    if not tracks:
        print("No liked tracks found (profile may be private).")
        return

    print_track_list(tracks)

    # ── Export CSV ─────────────────────────────────────────────────────────────
    if args.csv_path:
        export_csv(tracks, args.csv_path)

    # ── Download from YouTube ─────────────────────────────────────────────────
    if args.download:
        # Check yt-dlp + ffmpeg availability
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            print("[ERROR] yt-dlp not installed. Run:  uv sync")
            return

        import shutil
        if not shutil.which("ffmpeg"):
            print("[WARN] ffmpeg not found in PATH — mp3 conversion may fail.")
            print("       Install ffmpeg: https://ffmpeg.org/download.html\n")

        print(
            f"Downloading {len(tracks)} tracks from YouTube → {args.output}"
            f"  (max_duration={args.max_duration}s)\n"
        )
        ok, fail, skip = download_tracks(tracks, args.output, args.max_duration)
        print(f"\nDone: {ok} downloaded, {skip} skipped, {fail} failed")
        print(f"\nNext step:")
        print(f"  python src/encode.py --audio_dir {args.output} --encoder mert")
    else:
        print("Add --download to download audio from YouTube.")
        print(f"  python src/fetch_deezer.py {user_id} --download")


if __name__ == "__main__":
    main()
