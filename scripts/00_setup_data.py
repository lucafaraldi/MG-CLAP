#!/usr/bin/env python3
"""Download and organize AudioCaps + ESC-50 into the expected on-disk layout.

Usage:
    python scripts/00_setup_data.py --audiocaps --esc50

What it does:

* AudioCaps: pulls the official train/val/test CSVs from the audiocaps GitHub
  mirror, then uses yt-dlp to fetch the 10s YouTube clips called out by each
  row.  Default split is `val` (~495 rows). Train (~46k) and test (~975) take
  longer; pass `--audiocaps-split` to override.

* ESC-50: pulls the master zip from the official karolpiczak/ESC-50 release,
  unpacks `audio/` and `meta/` under data/esc50/.

YouTube downloads occasionally fail (videos go private). The script logs
failures and continues — the dataset loader skips missing files. We typically
recover ~90 % of AudioCaps val.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import requests  # certifi-backed; sidesteps macOS python.org missing-root-CA issue

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

AUDIOCAPS_CSVS = {
    # Use the cdjkim/audiocaps mirror — keeps the original column layout.
    "train": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv",
    "val":   "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv",
    "test":  "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv",
}

ESC50_ZIP = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def _venv_bin(name: str) -> str | None:
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else None


def _resolve_ytdlp() -> str:
    local = _venv_bin("yt-dlp")
    if local is not None:
        return local
    global_path = shutil.which("yt-dlp")
    if global_path is None:
        raise FileNotFoundError(
            "yt-dlp not found. Install it into the active venv or put it on PATH."
        )
    return global_path


def _resolve_ffmpeg() -> str | None:
    local = _venv_bin("ffmpeg")
    if local is not None:
        return local
    global_path = shutil.which("ffmpeg")
    if global_path is not None:
        return global_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# AudioCaps
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)


def _ytdlp(youtube_id: str, start: float, out_path: Path, duration: float = 10.0) -> bool:
    """Use yt-dlp + ffmpeg to grab a 10 s clip, mono 48 kHz wav."""
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    ytdlp_bin = _resolve_ytdlp()
    ffmpeg_bin = _resolve_ffmpeg()
    cmd = [
        ytdlp_bin,
        "--quiet",
        "--no-warnings",
        "-x",
        "--audio-format", "wav",
        "-o", str(out_path.with_suffix(".%(ext)s")),
        url,
    ]
    if ffmpeg_bin is not None:
        cmd.extend(["--ffmpeg-location", ffmpeg_bin])
        cmd.extend(
            [
                "--postprocessor-args",
                f"ffmpeg:-ss {start} -t {duration} -ac 1 -ar 48000",
            ]
        )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return out_path.exists() and result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def setup_audiocaps(splits: Iterable[str] = ("val",), max_workers: int = 4) -> None:
    base = DATA_DIR / "audiocaps"
    csv_dir = base / "csv"
    audio_dir = base / "audio"
    csv_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in AUDIOCAPS_CSVS:
            raise ValueError(f"unknown audiocaps split: {split}")
        csv_path = csv_dir / f"{split}.csv"
        if not csv_path.exists():
            _download(AUDIOCAPS_CSVS[split], csv_path)

        rows = []
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows.extend(reader)

        # Deduplicate at (youtube_id, start_time) — multiple captions per audio.
        unique_audio = {}
        for r in rows:
            yid = r.get("youtube_id") or r.get("ytid")
            start = float(r.get("start_time", 0.0))
            unique_audio.setdefault((yid, start), r)
        print(
            f"AudioCaps {split}: {len(rows)} caption rows, "
            f"{len(unique_audio)} unique audio clips"
        )

        out_split_dir = audio_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for (yid, start), r in unique_audio.items():
                out = out_split_dir / f"{yid}_{int(start)}.wav"
                futures[ex.submit(_ytdlp, yid, start, out)] = (yid, start)
            for i, fut in enumerate(as_completed(futures), start=1):
                if fut.result():
                    ok += 1
                else:
                    fail += 1
                if i % 25 == 0 or i == len(futures):
                    print(f"  [{split}] {i}/{len(futures)}  ok={ok}  fail={fail}")
        print(f"AudioCaps {split} done: ok={ok}  fail={fail}")


# ---------------------------------------------------------------------------
# ESC-50
# ---------------------------------------------------------------------------
def setup_esc50() -> None:
    base = DATA_DIR / "esc50"
    base.mkdir(parents=True, exist_ok=True)
    if (base / "meta" / "esc50.csv").exists() and (base / "audio").exists():
        n = sum(1 for _ in (base / "audio").glob("*.wav"))
        print(f"ESC-50 already present at {base} ({n} wavs)")
        return

    print(f"downloading ESC-50 zip... ({ESC50_ZIP})")
    r = requests.get(ESC50_ZIP, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    members = z.namelist()
    # Top-level dir is e.g. ESC-50-master/
    prefix = members[0].split("/")[0] + "/"
    for m in members:
        if not m.startswith(prefix):
            continue
        rel = m[len(prefix):]
        if not rel:
            continue
        target = base / rel
        if m.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"ESC-50 unpacked to {base}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audiocaps", action="store_true")
    p.add_argument(
        "--audiocaps-split",
        nargs="+",
        default=["val"],
        choices=["train", "val", "test"],
        help="Which AudioCaps splits to fetch. Default: val.",
    )
    p.add_argument("--esc50", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if not (args.audiocaps or args.esc50):
        p.error("pass at least one of --audiocaps / --esc50")

    if args.audiocaps:
        setup_audiocaps(args.audiocaps_split, max_workers=args.workers)
    if args.esc50:
        setup_esc50()


if __name__ == "__main__":
    main()
