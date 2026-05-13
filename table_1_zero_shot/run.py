#!/usr/bin/env python3
"""Table 1 — zero-shot ESC-50 accuracy under embedding shift.

Reproduces the paper's `Table_1_Implications_CLIP_Zero_Shot/shifting/shift_features.ipynb`
and `simulation/simulation.ipynb` for the audio modality:

* `--mode shift`     — load pre-extracted CLAP embeddings, apply a symmetric
                       shift along the gap direction Δ (computed on AudioCaps),
                       sweep λ ∈ [-1, +1], evaluate ESC-50 zero-shot accuracy
                       per 5-fold and overall.

* `--mode simulation` — pure synthetic experiment: for each target gap distance,
                        sample class-conditional audio + class prompts and
                        measure top-1 accuracy. Shows that varying the gap
                        directly varies retrieval accuracy.

Original paper finding (Table 1): for several CLIP zero-shot benchmarks,
INCREASING the gap (negative λ in our convention, i.e. moving the modalities
apart) improves accuracy by 1–6 points. The CLAP port asks: does the same
sign of effect hold for audio?

Run:
    python table_1_zero_shot/run.py --mode shift --backbone laion
    python table_1_zero_shot/run.py --mode simulation
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
    gap_distance, gap_vector, l2_normalize, shift_features,
)
from lib.synth import synth_class_prompts, synth_paired_embeddings  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def zero_shot_accuracy(
    audio_emb: np.ndarray, text_emb: np.ndarray, labels: np.ndarray
) -> float:
    """Top-1 accuracy via cosine similarity argmax. Embeddings should be L2-normalized."""
    sim = audio_emb @ text_emb.T  # (N, n_classes)
    pred = sim.argmax(axis=1)
    return float((pred == labels).mean())


def per_fold_accuracy(
    audio_emb: np.ndarray, text_emb: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> tuple[list[float], float]:
    accs = []
    for f in (1, 2, 3, 4, 5):
        m = folds == f
        if m.sum() == 0:
            continue
        accs.append(zero_shot_accuracy(audio_emb[m], text_emb, labels[m]))
    return accs, float(np.mean(accs)) if accs else 0.0


# ---------------------------------------------------------------------------
# Mode 1 — shift real CLAP embeddings and measure ESC-50 accuracy
# ---------------------------------------------------------------------------
def run_shift_mode(args) -> dict:
    audiocaps_cache = ROOT / "embeddings" / args.backbone / f"audiocaps__{args.split}.npz"
    esc50_cache = ROOT / "embeddings" / args.backbone / "esc50.npz"
    if not audiocaps_cache.exists() or not esc50_cache.exists():
        print(
            f"[warn] missing cache(s):\n  {audiocaps_cache}: {audiocaps_cache.exists()}"
            f"\n  {esc50_cache}: {esc50_cache.exists()}\n"
            "→ run scripts/01_extract_embeddings.py for both datasets, or use --mode simulation"
        )
        return {}

    ac = np.load(audiocaps_cache, allow_pickle=True)
    esc = np.load(esc50_cache, allow_pickle=True)
    audio_ac, text_ac = ac["audio"], ac["text"]
    audio_es, text_es = esc["audio"], esc["text"]
    labels = esc["labels"]; folds = esc["folds"]

    print(f"AudioCaps {args.split}: {audio_ac.shape[0]} pairs, dim={audio_ac.shape[1]}")
    print(f"ESC-50: {audio_es.shape[0]} clips, {text_es.shape[0]} class prompts")

    # Compute the gap direction on AudioCaps (the population that defines Δ)
    delta = gap_vector(audio_ac, text_ac, normalize=True)
    print(f"||Δ_audiocaps|| = {np.linalg.norm(delta):.4f}")

    # Baseline: λ = 0
    audio_n = l2_normalize(audio_es); text_n = l2_normalize(text_es)
    base_accs, base_mean = per_fold_accuracy(audio_n, text_n, labels, folds)
    print(f"baseline ESC-50 acc per fold: {[round(a, 3) for a in base_accs]}  mean={base_mean:.4f}")

    # Sweep λ
    lambdas = np.linspace(args.lam_min, args.lam_max, args.n_lambdas)
    rows = []
    for lam in lambdas:
        a_s, t_s = shift_features(audio_es, text_es, lam=float(lam), delta=delta)
        accs, mean = per_fold_accuracy(a_s, t_s, labels, folds)
        gap = gap_distance(a_s, t_s, normalize=False)
        rows.append({"lambda": float(lam), "gap": float(gap), "fold_accs": accs, "mean_acc": mean})
        print(f"  λ={lam:+.3f}  gap={gap:.3f}  acc={mean:.4f}")

    return {
        "lambdas": lambdas.tolist(),
        "gap_audiocaps": float(np.linalg.norm(delta)),
        "baseline_acc_per_fold": base_accs,
        "baseline_mean_acc": base_mean,
        "sweep": rows,
    }


# ---------------------------------------------------------------------------
# Mode 2 — pure synthetic simulation: gap distance vs zero-shot accuracy
# ---------------------------------------------------------------------------
def run_simulation_mode(args) -> dict:
    # Generate class-conditional audio + class prompts, shift the result, evaluate.
    audio, text, labels = synth_class_prompts(
        n_classes=args.n_classes,
        n_per_class=args.n_per_class,
        d=512,
        gap=args.gap_synth,
        class_separation=args.class_sep,
        seed=0,
    )

    # We have a fixed gap, no folds → use evenly-split synthetic folds for parity.
    folds = np.tile(np.arange(1, 6), int(np.ceil(audio.shape[0] / 5)))[: audio.shape[0]]

    delta = gap_vector(audio, text)  # use audio-text gap
    print(f"baseline synthetic gap: {np.linalg.norm(delta):.4f}")

    # Baseline
    base_accs, base_mean = per_fold_accuracy(l2_normalize(audio), l2_normalize(text), labels, folds)
    print(f"baseline acc: {base_mean:.4f} (per-fold: {[round(a, 3) for a in base_accs]})")

    lambdas = np.linspace(args.lam_min, args.lam_max, args.n_lambdas)
    rows = []
    for lam in lambdas:
        a_s, t_s = shift_features(audio, text, lam=float(lam), delta=delta)
        accs, mean = per_fold_accuracy(a_s, t_s, labels, folds)
        gap = gap_distance(a_s, t_s, normalize=False)
        rows.append({"lambda": float(lam), "gap": float(gap), "fold_accs": accs, "mean_acc": mean})
        print(f"  λ={lam:+.3f}  gap={gap:.3f}  acc={mean:.4f}")

    return {
        "lambdas": lambdas.tolist(),
        "gap_baseline": float(np.linalg.norm(delta)),
        "baseline_acc_per_fold": base_accs,
        "baseline_mean_acc": base_mean,
        "sweep": rows,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep(data: dict, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lambdas = [r["lambda"] for r in data["sweep"]]
    gaps = [r["gap"] for r in data["sweep"]]
    means = [r["mean_acc"] for r in data["sweep"]]
    fold_arr = np.array([r["fold_accs"] for r in data["sweep"]])

    axes[0].plot(lambdas, means, lw=2, color="#0ea5e9", marker="o", markersize=4)
    if fold_arr.size:
        axes[0].fill_between(
            lambdas, fold_arr.min(axis=1), fold_arr.max(axis=1),
            alpha=0.15, color="#0ea5e9", label="fold range",
        )
    base = data["baseline_mean_acc"]
    axes[0].axhline(base, color="black", lw=0.7, linestyle=":",
                    label=f"baseline (λ=0): {base:.3f}")
    axes[0].set_xlabel("shift λ"); axes[0].set_ylabel("zero-shot top-1 accuracy")
    axes[0].set_title("Accuracy vs shift λ")
    axes[0].legend(frameon=False)
    axes[0].axvline(0.0, color="black", lw=0.5, linestyle=":")

    axes[1].plot(gaps, means, lw=2, color="#10b981", marker="o", markersize=4)
    axes[1].set_xlabel("modality gap distance ‖Δ‖₂"); axes[1].set_ylabel("zero-shot top-1 accuracy")
    axes[1].set_title("Accuracy vs gap distance")
    axes[1].axvline(data.get("gap_audiocaps") or data.get("gap_baseline", 0),
                    color="black", lw=0.7, linestyle=":", label="baseline ‖Δ‖")
    axes[1].legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["shift", "simulation"], default="shift")
    p.add_argument("--backbone", choices=["laion", "msclap"], default="laion")
    p.add_argument("--split", default="val")
    p.add_argument("--lam-min", type=float, default=-1.0)
    p.add_argument("--lam-max", type=float, default=+1.0)
    p.add_argument("--n-lambdas", type=int, default=21)
    # simulation-only knobs
    p.add_argument("--n-classes", type=int, default=50)
    p.add_argument("--n-per-class", type=int, default=40)
    p.add_argument("--gap-synth", type=float, default=0.82)
    p.add_argument("--class-sep", type=float, default=1.5,
                   help="class-direction magnitude. Larger → easier zero-shot, smaller → harder.")
    args = p.parse_args()

    out_dir = ROOT / "results" / "table_1" / (args.mode if args.mode == "simulation" else f"{args.backbone}-shift")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "shift":
        data = run_shift_mode(args)
        if not data:
            return
        title = f"Table 1 — ESC-50 zero-shot under shift ({args.backbone})"
    else:
        data = run_simulation_mode(args)
        title = f"Table 1 (simulation) — synthetic gap sweep, {args.n_classes} classes"

    plot_sweep(data, out_dir / "shift_sweep.png", title=title)
    (out_dir / "summary.json").write_text(json.dumps(data, indent=2, default=float))
    print(f"\nresults under {out_dir}")


if __name__ == "__main__":
    main()
