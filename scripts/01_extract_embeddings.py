#!/usr/bin/env python3
"""Extract and cache CLAP audio + text embeddings.

This is the one piece every experiment needs. We cache to .npz so we never have
to re-run the (expensive) audio encoder.

Usage:
    python scripts/01_extract_embeddings.py --backbone laion --dataset audiocaps --split val
    python scripts/01_extract_embeddings.py --backbone msclap --dataset audiocaps --split val
    python scripts/01_extract_embeddings.py --backbone laion --dataset esc50
    python scripts/01_extract_embeddings.py --backbone msclap --dataset esc50

Output:
    embeddings/<backbone>/<dataset>__<split>.npz
        audio  : (N, d)  raw audio embeddings (NOT normalized)
        text   : (N, d)  raw text embeddings  (NOT normalized)
        labels : optional, classification target ids (ESC-50)
        meta   : json string with per-item metadata (paths, captions, ids)

We deliberately do not L2-normalize here so that downstream code (gap_utils,
shift, classification) controls normalization explicitly.

Resilience:
    * Files that fail header validation (empty / no audio stream / unreadable)
      are dropped BEFORE encoding, and the matched captions/labels are dropped
      with them.
    * Files that survive validation but error during encoding are wrapped in
      try/except and skipped, with the partner caption/label removed too.

A summary of dropped files is printed at the end of each run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.clap_models import load_backbone  # noqa: E402
from lib.datasets import AudioCaps, ESC50    # noqa: E402


def cache_path(backbone: str, dataset: str, split: str | None) -> Path:
    name = f"{dataset}__{split}" if split else dataset
    return ROOT / "embeddings" / backbone / f"{name}.npz"


# ---------------------------------------------------------------------------
def _validate_audio_paths(paths: list[str], min_frames: int = 100) -> tuple[list[int], list[str]]:
    """Return (kept_indices, reasons_for_drops). Uses soundfile.info — header-only,
    so very fast (a few thousand files per second)."""
    import soundfile as sf
    keep: list[int] = []
    reasons: list[str] = []
    for i, p in enumerate(tqdm(paths, desc="validating audio", unit="file")):
        try:
            info = sf.info(p)
            if info.frames < min_frames:
                reasons.append(f"{Path(p).name}: only {info.frames} frames")
                continue
        except Exception as e:
            reasons.append(f"{Path(p).name}: {type(e).__name__}: {e}")
            continue
        keep.append(i)
    return keep, reasons


def _encode_audio_resilient(
    backbone, paths: list[str], batch_size: int = 4,
) -> tuple[np.ndarray, list[int]]:
    """Encode audio with per-batch fallback to per-file on errors.

    Returns (embeddings, kept_indices_into_paths). embeddings.shape[0] == len(kept).
    """
    embs: list[np.ndarray] = []
    kept: list[int] = []
    failed: list[tuple[str, str]] = []

    # First pass: batched (fast path).
    i = 0
    pbar = tqdm(total=len(paths), desc="encoding audio", unit="clip")
    while i < len(paths):
        chunk = paths[i : i + batch_size]
        chunk_idx = list(range(i, i + len(chunk)))
        try:
            emb = backbone.encode_audio(chunk, batch_size=batch_size)
            embs.append(np.asarray(emb))
            kept.extend(chunk_idx)
        except Exception as batch_err:
            # Fall back to per-file so one bad clip in a batch doesn't drop the rest.
            for j, p in zip(chunk_idx, chunk):
                try:
                    emb = backbone.encode_audio([p], batch_size=1)
                    embs.append(np.asarray(emb))
                    kept.append(j)
                except Exception as e:
                    failed.append((p, f"{type(e).__name__}: {e}"))
        i += len(chunk)
        pbar.update(len(chunk))
    pbar.close()
    if failed:
        print(f"  [skip] {len(failed)} files failed to encode (e.g. {failed[0][1]})")

    if not embs:
        return np.zeros((0, 0)), []
    return np.concatenate(embs, axis=0), kept


# ---------------------------------------------------------------------------
def extract_audiocaps(backbone, split: str) -> dict:
    ds = AudioCaps(split=split, require_exists=True)
    if len(ds) == 0:
        raise RuntimeError(
            f"AudioCaps {split} is empty. Run scripts/00_setup_data.py first."
        )
    print(f"AudioCaps {split}: {len(ds)} items")
    paths = [str(p) for p in ds.audio_paths()]
    captions = ds.captions()

    print("  → validating audio files...")
    good_idx, reasons = _validate_audio_paths(paths)
    if reasons:
        print(f"  [skip] {len(reasons)} invalid before encoding")
        for r in reasons[:5]:
            print(f"      {r}")
        if len(reasons) > 5:
            print(f"      ... and {len(reasons) - 5} more")
    paths_ok = [paths[i] for i in good_idx]
    captions_ok = [captions[i] for i in good_idx]
    items_ok = [ds.items[i] for i in good_idx]
    print(f"  proceeding with {len(paths_ok)} / {len(paths)} clips")

    print("  → encoding audio...")
    audio_emb, kept = _encode_audio_resilient(backbone, paths_ok, batch_size=4)
    captions_kept = [captions_ok[i] for i in kept]
    items_kept = [items_ok[i] for i in kept]
    print(f"    audio: {audio_emb.shape}, dtype={audio_emb.dtype}")

    print("  → encoding text...")
    text_emb = backbone.encode_text(captions_kept, batch_size=32)
    print(f"    text:  {text_emb.shape}")

    meta = [
        {
            "audiocap_id": it.audiocap_id,
            "youtube_id": it.youtube_id,
            "start_time": it.start_time,
            "caption": it.caption,
            "audio_path": str(it.audio_path),
        }
        for it in items_kept
    ]
    return {"audio": audio_emb, "text": text_emb, "meta": json.dumps(meta)}


def extract_esc50(backbone) -> dict:
    ds = ESC50()
    print(f"ESC-50: {len(ds)} clips, {len(ds.class_names)} classes")
    paths_all = [str(it.audio_path) for it in ds.items]
    targets_all = np.array([it.target for it in ds.items], dtype=np.int64)
    folds_all = np.array([it.fold for it in ds.items], dtype=np.int64)

    print("  → validating audio files...")
    good_idx, reasons = _validate_audio_paths(paths_all)
    if reasons:
        print(f"  [skip] {len(reasons)} invalid before encoding")
        for r in reasons[:5]:
            print(f"      {r}")
    paths_ok = [paths_all[i] for i in good_idx]
    targets_ok = targets_all[good_idx]
    folds_ok = folds_all[good_idx]
    print(f"  proceeding with {len(paths_ok)} / {len(paths_all)} clips")

    prompts = ds.class_prompts()
    print("  → encoding audio...")
    audio_emb, kept = _encode_audio_resilient(backbone, paths_ok, batch_size=4)
    targets_kept = targets_ok[kept]
    folds_kept = folds_ok[kept]
    print(f"    audio: {audio_emb.shape}")
    print("  → encoding class prompts...")
    text_emb = backbone.encode_text(prompts, batch_size=32)
    print(f"    text:  {text_emb.shape}")

    meta = {
        "class_names": ds.class_names,
        "prompts": prompts,
        "filenames": [Path(paths_ok[i]).name for i in kept],
    }
    return {
        "audio": audio_emb,
        "text": text_emb,
        "labels": targets_kept,
        "folds": folds_kept,
        "meta": json.dumps(meta),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone", choices=["laion", "msclap"], required=True,
    )
    p.add_argument(
        "--dataset", choices=["audiocaps", "esc50"], required=True,
    )
    p.add_argument(
        "--split", default="val",
        choices=["train", "val", "test"],
        help="(AudioCaps only) which split. Default: val.",
    )
    p.add_argument("--force", action="store_true", help="overwrite cache")
    args = p.parse_args()

    split = args.split if args.dataset == "audiocaps" else None
    out = cache_path(args.backbone, args.dataset, split)
    if out.exists() and not args.force:
        print(f"cache exists: {out} (use --force to overwrite)")
        return

    print(f"loading backbone: {args.backbone}")
    backbone = load_backbone(args.backbone)
    print(f"  info: {backbone.info}")

    if args.dataset == "audiocaps":
        payload = extract_audiocaps(backbone, args.split)
    else:
        payload = extract_esc50(backbone)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    sz_mb = out.stat().st_size / 1e6
    print(f"\nsaved {out} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
