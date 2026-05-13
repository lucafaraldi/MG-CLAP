#!/usr/bin/env python3
"""Figure 2c — pretrained-encoder variant.

Counterpart of the original repo's
`Figure_2c_scatter_cones_random_init/real_data_ImageNet_pretrained/`: instead
of feeding real data through a randomly-initialized encoder and measuring the
cone, we feed real audio + text through the PRETRAINED CLAP encoders (both
LAION-CLAP and MS-CLAP) and verify that the cone phenomenon survives training.

Output: a single scatter showing up to four cones — LAION audio, LAION text,
MS-CLAP audio, MS-CLAP text — projected jointly with PCA and UMAP. If the gap
analysis is right we should see four distinct, narrow clusters.

This script is purely a re-projection of the already-cached embeddings, so it
runs in milliseconds once `scripts/01_extract_embeddings.py` has been run for
both backbones.

Run:
    python figure_2_cone_effect/2c_scatter_cones/run_pretrained.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.gap_utils import (  # noqa: E402
    l2_normalize, intra_modal_avg_cosine, gap_distance,
)
from lib.viz import pca_2d, umap_2d  # noqa: E402


BACKBONES = ("laion", "msclap")


def load(backbone: str, split: str, max_n: int):
    cache = ROOT / "embeddings" / backbone / f"audiocaps__{split}.npz"
    if not cache.exists():
        return None
    data = np.load(cache, allow_pickle=True)
    audio = l2_normalize(data["audio"])
    text = l2_normalize(data["text"])
    if audio.shape[0] > max_n:
        rng = np.random.default_rng(0)
        idx = rng.choice(audio.shape[0], max_n, replace=False)
        audio, text = audio[idx], text[idx]
    return audio, text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--max-n", type=int, default=1000)
    args = p.parse_args()

    out_dir = ROOT / "results" / "figure_2" / "2c_scatter_cones_pretrained"
    out_dir.mkdir(parents=True, exist_ok=True)

    cones: list[tuple[str, np.ndarray]] = []
    summary: dict = {}
    for bb in BACKBONES:
        pair = load(bb, args.split, args.max_n)
        if pair is None:
            print(f"[skip] no cache for {bb}; run scripts/01_extract_embeddings.py first")
            continue
        a, t = pair
        cones.append((f"{bb} audio", a))
        cones.append((f"{bb} text", t))
        summary[bb] = {
            "n_pairs": int(a.shape[0]),
            "audio_cone": float(intra_modal_avg_cosine(a)),
            "text_cone": float(intra_modal_avg_cosine(t)),
            "gap_distance": float(gap_distance(a, t, normalize=False)),
        }

    if not cones:
        print("no caches found; nothing to plot")
        return

    print(json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    labels = [c[0] for c in cones]
    embs = [c[1] for c in cones]
    palette = plt.cm.tab10(np.linspace(0, 1, len(cones)))

    pca = pca_2d(*embs)
    ums = umap_2d(*embs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for (lbl, xy_pca, xy_umap, color) in zip(labels, pca, ums, palette):
        for ax, xy in zip(axes, (xy_pca, xy_umap)):
            ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.5, c=[color], label=lbl, edgecolor="none")
            ax.scatter(xy[:, 0].mean(), xy[:, 1].mean(), s=240, marker="X",
                       c=[color], edgecolor="black", linewidth=1.0, zorder=5)
    axes[0].set_title("PCA — pretrained CLAP cones")
    axes[1].set_title("UMAP — pretrained CLAP cones")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Figure 2c (pretrained) — cones persist after training")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_cones_pretrained.png", dpi=150)
    plt.close(fig)
    print(f"\nresults under {out_dir}")


if __name__ == "__main__":
    main()
