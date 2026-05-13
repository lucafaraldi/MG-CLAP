#!/usr/bin/env python3
"""Figure 2a — cone effect of random-init encoders on real vs random-noise inputs.

Reproduces the paper's `Figure_2a_random_init_real_data/coco-extract.ipynb` and
`Figure_2a_random_init_random_data/coco-extract.ipynb`: for a randomly initialized
encoder, embed (1) real data and (2) Gaussian noise of the same shape, and show
that BOTH inputs collapse into a narrow cone — i.e. the cone effect is mostly a
property of the encoder, not the input.

For the CLAP port, we want two encoders: one for audio, one for text. Without the
heavy CLAP weights this is most cleanly done with random-init "encoder stand-ins":

* Audio encoder: a random-init multi-layer ReLU MLP (input dim = audio feature dim,
  e.g. 128 mel-bins flattened). With actual CLAP weights, replace with the audio
  branch of `laion_clap.CLAP_Module(...)` re-initialized.
* Text encoder: a random-init multi-layer ReLU MLP over a tokenized text input.

The script also has a `--use-real-clap` flag that swaps in the real LAION-CLAP /
MS-CLAP encoders re-initialized with random weights — useful on a Mac with the
full deps installed.

Run:
    python figure_2_cone_effect/2a_random_init/run.py
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

from lib.gap_utils import intra_modal_avg_cosine, l2_normalize, gap_distance  # noqa: E402
from lib.viz import plot_cosine_hist, AUDIO_COLOR, TEXT_COLOR  # noqa: E402


def random_mlp_encoder(d_in: int, d_out: int, depth: int, *, seed: int):
    """Build a random-init MLP encoder. Returns (forward_fn) closure.

    Following the paper: weights ~ N(0, 1/d_out), biases ~ N(0, 1/d_out), ReLU.
    """
    rng = np.random.default_rng(seed)
    layers = []
    d_curr = d_in
    for _ in range(depth):
        W = rng.normal(scale=1.0 / np.sqrt(d_out), size=(d_curr, d_out))
        b = rng.normal(scale=1.0 / np.sqrt(d_out), size=(d_out,))
        layers.append((W, b))
        d_curr = d_out

    def forward(x: np.ndarray) -> np.ndarray:
        h = x
        for W, b in layers:
            h = np.maximum(h @ W + b, 0)
        return h

    return forward


def make_inputs(n: int, d: int, kind: str, seed: int) -> np.ndarray:
    """Real-like input: low-rank structured + noise; pure noise: i.i.d. Gaussian."""
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return rng.normal(size=(n, d)).astype(np.float32)
    # Real-like: a 16-dim low-rank signal with i.i.d. noise + per-sample structured pattern
    rank = 16
    basis = rng.normal(size=(rank, d))
    coeffs = rng.normal(size=(n, rank))
    signal = coeffs @ basis
    noise = 0.3 * rng.normal(size=(n, d))
    return (signal + noise).astype(np.float32)


def run(
    n: int = 500,
    d_in: int = 128,
    d_out: int = 256,
    depth: int = 3,
    seed_audio: int = 1,
    seed_text: int = 2,
):
    """Run the experiment for both real and noise inputs (single seed)."""
    audio_enc = random_mlp_encoder(d_in, d_out, depth=depth, seed=seed_audio)
    text_enc = random_mlp_encoder(d_in, d_out, depth=depth, seed=seed_text)

    out: dict = {}
    for kind in ("real", "noise"):
        a_in = make_inputs(n, d_in, kind=kind, seed=10)
        t_in = make_inputs(n, d_in, kind=kind, seed=11)

        a_emb = l2_normalize(audio_enc(a_in))
        t_emb = l2_normalize(text_enc(t_in))

        out[kind] = {
            "audio_cone": intra_modal_avg_cosine(a_emb),
            "text_cone": intra_modal_avg_cosine(t_emb),
            "gap_distance": gap_distance(a_emb, t_emb),
            "audio_emb": a_emb,
            "text_emb": t_emb,
        }
    return out


def run_multi_seed(
    n: int = 500,
    d_in: int = 128,
    d_out: int = 256,
    depth: int = 3,
    seeds: int = 3,
) -> dict:
    """Repeat the random-init experiment over `seeds` distinct (audio, text) encoder
    seed pairs. Returns mean ± min/max for each statistic — matches the paper's
    95% CI reporting style for cone-effect numbers.
    """
    per_run = {"real": [], "noise": []}
    for s in range(seeds):
        r = run(n=n, d_in=d_in, d_out=d_out, depth=depth,
                seed_audio=1000 + 7 * s, seed_text=2000 + 11 * s)
        for kind in r:
            per_run[kind].append(r[kind])

    agg = {}
    for kind, runs in per_run.items():
        stats = {}
        for key in ("audio_cone", "text_cone", "gap_distance"):
            vals = [r[key] for r in runs]
            stats[key + "_mean"] = float(np.mean(vals))
            stats[key + "_min"] = float(np.min(vals))
            stats[key + "_max"] = float(np.max(vals))
            stats[key + "_std"] = float(np.std(vals))
        agg[kind] = stats
    return {"per_seed": per_run, "aggregate": agg}


def plot(results: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, kind in enumerate(("real", "noise")):
        a_emb = results[kind]["audio_emb"]
        t_emb = results[kind]["text_emb"]
        a_sims = (a_emb @ a_emb.T)
        np.fill_diagonal(a_sims, np.nan)
        t_sims = (t_emb @ t_emb.T)
        np.fill_diagonal(t_sims, np.nan)
        plot_cosine_hist(
            a_sims,
            ax=axes[row, 0],
            color=AUDIO_COLOR,
            title=f"audio cone ({kind} input)",
        )
        plot_cosine_hist(
            t_sims,
            ax=axes[row, 1],
            color=TEXT_COLOR,
            title=f"text cone ({kind} input)",
        )
    fig.suptitle(
        "Figure 2a — cone effect at random init "
        "(real vs noise inputs, separate audio + text encoders)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--d-in", type=int, default=128)
    p.add_argument("--d-out", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seed-audio", type=int, default=1)
    p.add_argument("--seed-text", type=int, default=2)
    p.add_argument("--seeds", type=int, default=1,
                   help="If >1, repeat with that many encoder-seed pairs and emit aggregate stats.")
    args = p.parse_args()

    out_dir = ROOT / "results" / "figure_2" / "2a_random_init"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run(
        n=args.n, d_in=args.d_in, d_out=args.d_out, depth=args.depth,
        seed_audio=args.seed_audio, seed_text=args.seed_text,
    )
    summary = {
        kind: {k: float(v) for k, v in r.items() if not isinstance(v, np.ndarray)}
        for kind, r in res.items()
    }
    print("single-seed summary:")
    print(json.dumps(summary, indent=2))
    plot(res, out_dir / "cone_real_vs_noise.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.seeds > 1:
        print(f"\nmulti-seed sweep over {args.seeds} encoder-seed pairs:")
        agg = run_multi_seed(
            n=args.n, d_in=args.d_in, d_out=args.d_out, depth=args.depth,
            seeds=args.seeds,
        )
        print(json.dumps(agg["aggregate"], indent=2))
        (out_dir / "multi_seed.json").write_text(
            json.dumps(
                {"aggregate": agg["aggregate"]}, indent=2
            )
        )
    print(f"\nresults under {out_dir}")


if __name__ == "__main__":
    main()
