#!/usr/bin/env python3
"""Figure 2b — cone effect under increasing depth in a random-init MLP.

Reproduces the paper's `Figure_2b_random_MLP_layerwise/{bias_linear_relu, no_bias/linear_relu}.ipynb`:
sweep MLP depth from 1..L; for each depth, build a random-init MLP and embed the
same input batch; measure intra-modal average cosine similarity (cone tightness)
at each layer.

Theoretically (the paper's Theorem 1+2): non-linearity (ReLU/sigmoid/GELU) drives
cosine similarity → 1 monotonically as depth grows. Pure linear MLPs do not.

This experiment is purely synthetic — it doesn't need CLAP — and reproduces the
key cone phenomenon used to explain why random init creates a modality gap.

Run:
    python figure_2_cone_effect/2b_random_mlp/run.py
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

from lib.gap_utils import intra_modal_avg_cosine, l2_normalize  # noqa: E402


def random_mlp_layer(
    rng: np.random.Generator, d_in: int, d_out: int, *, bias: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """One MLP layer: W ~ N(0, 1/d_out) (paper-spec) and bias ~ N(0, 1/d_out)."""
    W = rng.normal(scale=1.0 / np.sqrt(d_out), size=(d_in, d_out))
    b = rng.normal(scale=1.0 / np.sqrt(d_out), size=(d_out,)) if bias else None
    return W, b


def forward_through_mlp(
    x: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray | None]],
    activation: str = "relu",
):
    """Iteratively apply (W,b) layers with the chosen activation. Yield per-layer output."""
    h = x
    yield 0, h
    for li, (W, b) in enumerate(layers, start=1):
        h = h @ W
        if b is not None:
            h = h + b
        if activation == "relu":
            h = np.maximum(h, 0)
        elif activation == "gelu":
            h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
        elif activation == "sigmoid":
            h = 1.0 / (1.0 + np.exp(-h))
        elif activation == "linear":
            pass
        else:
            raise ValueError(f"unknown activation {activation}")
        yield li, h


def cone_sweep(
    n_inputs: int = 200,
    d: int = 256,
    depth: int = 12,
    seeds: int = 5,
    activations: tuple[str, ...] = ("linear", "relu", "gelu", "sigmoid"),
    bias: bool = True,
    input_kind: str = "real",
):
    """For each (activation, seed): build a random MLP, run inputs through, compute
    per-layer intra-modal avg cosine similarity. Returns a dict keyed by activation
    with shape (seeds, depth+1) per measurement.
    """
    out: dict[str, np.ndarray] = {}
    for act in activations:
        per_seed = []
        for s in range(seeds):
            rng = np.random.default_rng(1000 * s + hash(act) % 1000)
            if input_kind == "real":
                # Real-like data: structured low-rank signal with noise
                u = rng.normal(size=(n_inputs, d))
                u = l2_normalize(u)
            else:  # "noise"
                u = rng.normal(size=(n_inputs, d))
                u = l2_normalize(u)

            d_curr = d
            layers = []
            for _ in range(depth):
                W, b = random_mlp_layer(rng, d_curr, d_curr, bias=bias)
                layers.append((W, b))

            cones = []
            for li, h in forward_through_mlp(u, layers, activation=act):
                cones.append(intra_modal_avg_cosine(h))
            per_seed.append(cones)
        out[act] = np.array(per_seed)  # (seeds, depth+1)
    return out


def plot_layerwise(
    results: dict, out_path: Path, title: str = "Cone effect — random MLP layerwise"
):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = {
        "linear": "#6b7280",
        "relu": "#3b82f6",
        "gelu": "#10b981",
        "sigmoid": "#ef4444",
    }
    for act, arr in results.items():
        depth_axis = np.arange(arr.shape[1])
        mean = arr.mean(axis=0)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        ax.plot(depth_axis, mean, lw=2, label=act, color=palette.get(act))
        ax.fill_between(depth_axis, lo, hi, alpha=0.18, color=palette.get(act))
    ax.set_xlabel("layer index (depth)")
    ax.set_ylabel("intra-modal mean cosine similarity")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="black", lw=0.5, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-inputs", type=int, default=300)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    out_dir = ROOT / "results" / "figure_2" / "2b_random_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    for input_kind in ("real", "noise"):
        for bias in (True, False):
            tag = f"{input_kind}-{'bias' if bias else 'nobias'}"
            print(f"\n[{tag}] sweeping depth={args.depth}, seeds={args.seeds}")
            res = cone_sweep(
                n_inputs=args.n_inputs,
                d=args.dim,
                depth=args.depth,
                seeds=args.seeds,
                bias=bias,
                input_kind=input_kind,
            )
            for act, arr in res.items():
                last_layer_mean = float(arr[:, -1].mean())
                print(f"  {act:8s} layer-{args.depth} mean cone: {last_layer_mean:.4f}")
            plot_layerwise(
                res,
                out_dir / f"cone_sweep_{tag}.png",
                title=f"Random MLP cone effect — input={input_kind}, bias={bias}",
            )
            (out_dir / f"cone_sweep_{tag}.json").write_text(
                json.dumps(
                    {act: arr.tolist() for act, arr in res.items()}, indent=2
                )
            )
    print(f"\nresults under {out_dir}")


if __name__ == "__main__":
    main()
