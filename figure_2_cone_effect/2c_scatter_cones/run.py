#!/usr/bin/env python3
"""Figure 2c — many random inits → many distinct cones → automatic modality gap.

Reproduces the paper's `Figure_2c_scatter_cones_random_init/` notebooks: for K
different random initializations of an encoder, embed the same input batch and
project everything jointly with PCA/UMAP. We expect to see K narrow, separated
clusters — visual evidence that random init creates the gap structure even
before any contrastive training.

For the CLAP port, this is the cleanest way to show the "two encoders → two
cones" intuition: pick two random inits, treat one as "audio encoder" and the
other as "text encoder", and measure the gap that pops out for free.

Run:
    python figure_2_cone_effect/2c_scatter_cones/run.py
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
    l2_normalize, intra_modal_avg_cosine, gap_distance, modality_centroids,
)
from lib.viz import pca_2d, umap_2d  # noqa: E402


def random_mlp_encoder(d_in: int, d_out: int, depth: int, seed: int):
    rng = np.random.default_rng(seed)
    layers = []
    d_curr = d_in
    for _ in range(depth):
        W = rng.normal(scale=1.0 / np.sqrt(d_out), size=(d_curr, d_out))
        b = rng.normal(scale=1.0 / np.sqrt(d_out), size=(d_out,))
        layers.append((W, b))
        d_curr = d_out

    def fwd(x):
        h = x
        for W, b in layers:
            h = np.maximum(h @ W + b, 0)
        return h
    return fwd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300, help="inputs per encoder")
    p.add_argument("--d-in", type=int, default=128)
    p.add_argument("--d-out", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n-encoders", type=int, default=8)
    p.add_argument("--input-kind", choices=("real", "noise"), default="real")
    args = p.parse_args()

    out_dir = ROOT / "results" / "figure_2" / "2c_scatter_cones"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    if args.input_kind == "real":
        rank = 16
        basis = rng.normal(size=(rank, args.d_in))
        coeffs = rng.normal(size=(args.n, rank))
        x = (coeffs @ basis + 0.3 * rng.normal(size=(args.n, args.d_in))).astype(np.float32)
    else:
        x = rng.normal(size=(args.n, args.d_in)).astype(np.float32)

    embeds: list[np.ndarray] = []
    cones: list[float] = []
    for k in range(args.n_encoders):
        enc = random_mlp_encoder(args.d_in, args.d_out, depth=args.depth, seed=100 + k)
        emb = l2_normalize(enc(x))
        embeds.append(emb)
        cones.append(intra_modal_avg_cosine(emb))

    # Pairwise gap distances between encoder cones (a measure of how separated they are).
    n_enc = len(embeds)
    pairwise_gap = np.zeros((n_enc, n_enc))
    for i in range(n_enc):
        for j in range(n_enc):
            if i != j:
                pairwise_gap[i, j] = gap_distance(embeds[i], embeds[j], normalize=False)

    summary = {
        "n_encoders": n_enc,
        "input_kind": args.input_kind,
        "per_encoder_cone_mean": [float(c) for c in cones],
        "pairwise_gap_min": float(pairwise_gap[pairwise_gap > 0].min()),
        "pairwise_gap_max": float(pairwise_gap.max()),
        "pairwise_gap_mean": float(
            pairwise_gap.sum() / max(1, n_enc * (n_enc - 1))
        ),
    }
    print(json.dumps(summary, indent=2))

    # ----- PCA + UMAP joint scatter -----
    palette = plt.cm.tab10(np.linspace(0, 1, n_enc))
    a_xy = pca_2d(*embeds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k, xy in enumerate(a_xy):
        axes[0].scatter(
            xy[:, 0], xy[:, 1], s=8, alpha=0.7, c=[palette[k]], label=f"enc {k}",
            edgecolor="none",
        )
        axes[0].scatter(
            xy[:, 0].mean(), xy[:, 1].mean(),
            s=200, marker="X", c=[palette[k]], edgecolor="black", linewidth=1.0, zorder=5,
        )
    axes[0].set_title("PCA — joint")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    u_xy = umap_2d(*embeds)
    for k, xy in enumerate(u_xy):
        axes[1].scatter(
            xy[:, 0], xy[:, 1], s=8, alpha=0.7, c=[palette[k]],
            edgecolor="none",
        )
        axes[1].scatter(
            xy[:, 0].mean(), xy[:, 1].mean(),
            s=200, marker="X", c=[palette[k]], edgecolor="black", linewidth=1.0, zorder=5,
        )
    axes[1].set_title("UMAP — joint")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle(
        f"Figure 2c — {n_enc} random-init encoders → {n_enc} cones "
        f"(input={args.input_kind})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_cones.png", dpi=150)
    plt.close(fig)

    # ----- pairwise gap heatmap -----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(pairwise_gap, cmap="magma")
    ax.set_xlabel("encoder j"); ax.set_ylabel("encoder i")
    ax.set_title("Pairwise gap ‖Δ_ij‖₂  (random-init encoders)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_gap.png", dpi=150)
    plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nresults under {out_dir}")


if __name__ == "__main__":
    main()
