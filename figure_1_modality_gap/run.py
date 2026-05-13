#!/usr/bin/env python3
"""Figure 1 — visualize the modality gap.

Reproduces, for CLAP, what the original repo's `Figure_1_Modality_Gap/` does for
CLIP / CLASP / ConVIRT / VideoCLIP: compute and visualize the gap between
audio and text embeddings on a paired dataset (AudioCaps).

Per backbone (`laion` or `msclap`) we report:

* Gap distance ‖Δ‖₂ on L2-normalized embeddings
* Modality centroids (location and norm)
* Mean pair cosine similarity (audio_i · text_i)
* Intra-modal cone statistics (avg pairwise cosine inside each modality)

And produce three figures:

* `gap_pca.png`   — joint PCA of audio + text with centroids
* `gap_umap.png`  — joint UMAP, same idea
* `gap_cosines.png` — histograms of pair cosines + intra-modal cosines

Run:
    python figure_1_modality_gap/run.py --backbone laion
    python figure_1_modality_gap/run.py --backbone msclap

Or with synthetic data (no CLAP install needed) to test the pipeline:
    python figure_1_modality_gap/run.py --synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.gap_utils import (  # noqa: E402
    l2_normalize, gap_vector, gap_distance, modality_centroids,
    cosine_sim_matrix, intra_modal_avg_cosine,
)
from lib.viz import plot_modality_2d, pca_2d, umap_2d, AUDIO_COLOR, TEXT_COLOR  # noqa: E402
from lib.synth import synth_paired_embeddings  # noqa: E402


def load_audiocaps_embeddings(backbone: str, split: str = "val"):
    cache = ROOT / "embeddings" / backbone / f"audiocaps__{split}.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"No cached embeddings: {cache}.\n"
            f"Run: python scripts/01_extract_embeddings.py "
            f"--backbone {backbone} --dataset audiocaps --split {split}"
        )
    data = np.load(cache, allow_pickle=True)
    return data["audio"], data["text"]


def compute_stats(audio_raw: np.ndarray, text_raw: np.ndarray) -> dict:
    audio = l2_normalize(audio_raw)
    text = l2_normalize(text_raw)
    n = audio.shape[0]
    delta = gap_vector(audio_raw, text_raw)
    a_c, t_c = modality_centroids(audio_raw, text_raw)
    pair_cos = (audio * text).sum(axis=-1)
    return {
        "n_pairs": int(n),
        "embed_dim": int(audio.shape[1]),
        "gap_distance": float(np.linalg.norm(delta)),
        "audio_centroid_norm": float(np.linalg.norm(a_c)),
        "text_centroid_norm": float(np.linalg.norm(t_c)),
        "pair_cosine_mean": float(pair_cos.mean()),
        "pair_cosine_std": float(pair_cos.std()),
        "audio_cone_mean": float(intra_modal_avg_cosine(audio)),
        "text_cone_mean": float(intra_modal_avg_cosine(text)),
    }


def plot_pca(audio: np.ndarray, text: np.ndarray, out_path: Path, title: str):
    a_xy, t_xy = pca_2d(l2_normalize(audio), l2_normalize(text))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_modality_2d(a_xy, t_xy, title=title, show_pairs=80, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_umap(audio: np.ndarray, text: np.ndarray, out_path: Path, title: str):
    a_xy, t_xy = umap_2d(l2_normalize(audio), l2_normalize(text))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_modality_2d(a_xy, t_xy, title=title, show_pairs=80, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cosines(audio: np.ndarray, text: np.ndarray, out_path: Path, title: str):
    audio = l2_normalize(audio); text = l2_normalize(text)
    pair_cos = (audio * text).sum(axis=-1)
    aa = cosine_sim_matrix(audio); np.fill_diagonal(aa, np.nan)
    tt = cosine_sim_matrix(text); np.fill_diagonal(tt, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(pair_cos, bins=50, color="#10b981", alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[0].axvline(pair_cos.mean(), color="black", lw=1, linestyle="--",
                    label=f"mean={pair_cos.mean():.3f}")
    axes[0].set_title("aligned pair cos(audio_i, text_i)")
    axes[0].set_xlabel("cosine similarity"); axes[0].legend(frameon=False)

    axes[1].hist(aa[~np.isnan(aa)], bins=50, color=AUDIO_COLOR, alpha=0.85,
                 edgecolor="white", linewidth=0.4)
    am = float(np.nanmean(aa))
    axes[1].axvline(am, color="black", lw=1, linestyle="--", label=f"mean={am:.3f}")
    axes[1].set_title("intra-audio cosines (cone)")
    axes[1].set_xlabel("cosine similarity"); axes[1].legend(frameon=False)

    axes[2].hist(tt[~np.isnan(tt)], bins=50, color=TEXT_COLOR, alpha=0.85,
                 edgecolor="white", linewidth=0.4)
    tm = float(np.nanmean(tt))
    axes[2].axvline(tm, color="black", lw=1, linestyle="--", label=f"mean={tm:.3f}")
    axes[2].set_title("intra-text cosines (cone)")
    axes[2].set_xlabel("cosine similarity"); axes[2].legend(frameon=False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["laion", "msclap"], default="laion")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic embeddings (skip CLAP) to test the pipeline.")
    p.add_argument("--n-synth", type=int, default=500)
    p.add_argument("--gap-synth", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-points", type=int, default=2000,
                   help="Subsample for UMAP/scatter plots if dataset is huge.")
    args = p.parse_args()

    if args.synthetic:
        print(f"[synthetic] generating n={args.n_synth} pairs with target gap={args.gap_synth}")
        audio, text = synth_paired_embeddings(
            n=args.n_synth, d=512, gap=args.gap_synth, seed=args.seed,
        )
        tag = f"synthetic-gap{args.gap_synth}"
        title = f"Figure 1 — synthetic (target gap {args.gap_synth})"
    else:
        print(f"[real] loading {args.backbone} on AudioCaps {args.split}")
        audio, text = load_audiocaps_embeddings(args.backbone, split=args.split)
        tag = f"{args.backbone}-audiocaps-{args.split}"
        title = f"Figure 1 — {args.backbone} on AudioCaps {args.split}"

    if audio.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(audio.shape[0], args.max_points, replace=False)
        audio, text = audio[idx], text[idx]

    stats = compute_stats(audio, text)
    print(json.dumps(stats, indent=2))

    out_dir = ROOT / "results" / "figure_1" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pca(audio, text, out_dir / "gap_pca.png", title=title + " — PCA")
    plot_umap(audio, text, out_dir / "gap_umap.png", title=title + " — UMAP")
    plot_cosines(audio, text, out_dir / "gap_cosines.png", title=title)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nfigures + stats.json saved under {out_dir}")


if __name__ == "__main__":
    main()
