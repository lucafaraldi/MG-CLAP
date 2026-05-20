#!/usr/bin/env python3
"""Figure 3 — contrastive loss landscape vs modality gap distance and temperature.

Reproduces the four notebooks under `Figure_3_Contrastive_Learning/` from the
original repo:

    3a  get_gap_stats          → compute Δ, ‖Δ‖, centroids on real embeddings
    3b  mismatched_simulation  → sweep gap distance, evaluate InfoNCE for many τ
    3c  3d_sphere              → 3D-sphere visualization of two cones at varying gap
    3d  plot_optimization_exp  → simulate optimizing the gap under InfoNCE

Key finding (paper, Sec. 4.3): at the typical CLIP temperature τ ≈ 0.01 the
InfoNCE loss has a NON-ZERO minimum gap distance. Increasing τ pushes the
minimum towards zero, allowing the model to close the gap. This is the
mechanistic explanation for why CLIP maintains a gap during training.

For the CLAP port, we expect the same picture: the minimum-loss gap depends on τ
in the same way, regardless of modality. Run:

    python figure_3_contrastive_learning/run.py
    python figure_3_contrastive_learning/run.py --backbone laion --real
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
    gap_distance, gap_vector, info_nce_loss, l2_normalize, shift_features,
)
from lib.provenance import command_string, file_info, get_git_commit, machine_info, utc_timestamp  # noqa: E402
from lib.synth import synth_paired_embeddings  # noqa: E402


# ---------------------------------------------------------------------------
# 3a — Gap stats on real CLAP embeddings (or synthetic fallback)
# ---------------------------------------------------------------------------
def gap_stats_on_data(audio: np.ndarray, text: np.ndarray) -> dict:
    delta = gap_vector(audio, text)
    return {
        "n_pairs": int(audio.shape[0]),
        "embed_dim": int(audio.shape[1]),
        "gap_distance": float(np.linalg.norm(delta)),
        "audio_centroid_norm": float(np.linalg.norm(l2_normalize(audio).mean(0))),
        "text_centroid_norm": float(np.linalg.norm(l2_normalize(text).mean(0))),
        "delta_top5_axes": np.argsort(-np.abs(delta))[:5].tolist(),
        "delta_top5_magnitudes": np.sort(np.abs(delta))[-5:][::-1].tolist(),
    }


# ---------------------------------------------------------------------------
# 3b — InfoNCE vs gap distance, swept across temperature τ
# ---------------------------------------------------------------------------
def loss_landscape_sweep(
    audio: np.ndarray,
    text: np.ndarray,
    temperatures: tuple[float, ...],
    lambdas: np.ndarray,
) -> dict:
    """For each (τ, λ): apply symmetric shift, compute InfoNCE, record (gap, loss)."""
    audio = l2_normalize(audio)
    text = l2_normalize(text)
    delta = gap_vector(audio, text, normalize=False)

    out: dict[float, dict] = {}
    for tau in temperatures:
        gaps = []
        losses = []
        for lam in lambdas:
            a_shift, t_shift = shift_features(audio, text, lam=float(lam), delta=delta)
            gaps.append(gap_distance(a_shift, t_shift, normalize=False))
            losses.append(info_nce_loss(a_shift, t_shift, temperature=float(tau)))
        out[float(tau)] = {
            "lambdas": lambdas.tolist(),
            "gaps": gaps,
            "losses": losses,
        }
    return out


# ---------------------------------------------------------------------------
# 3c — 3D-sphere visualization
# ---------------------------------------------------------------------------
def sphere_viz(audio: np.ndarray, text: np.ndarray, lambdas: tuple[float, ...], out_path: Path):
    """Project audio + text onto the leading-3 PCA axes and scatter on the unit sphere."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    audio = l2_normalize(audio); text = l2_normalize(text)
    stacked = np.concatenate([audio, text], axis=0)
    # Centred PCA
    mean = stacked.mean(0, keepdims=True)
    cov = (stacked - mean).T @ (stacked - mean) / max(1, stacked.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pcs = eigvecs[:, -3:][:, ::-1]   # top-3 components

    fig = plt.figure(figsize=(4 * len(lambdas), 4))
    for i, lam in enumerate(lambdas):
        a_s, t_s = shift_features(audio, text, lam=lam)
        a_xyz = l2_normalize(a_s @ pcs)
        t_xyz = l2_normalize(t_s @ pcs)
        ax = fig.add_subplot(1, len(lambdas), i + 1, projection="3d")
        # Translucent unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(sx, sy, sz, alpha=0.06, color="grey", edgecolor="none")
        ax.scatter(a_xyz[:, 0], a_xyz[:, 1], a_xyz[:, 2], s=6, c="#3b82f6", alpha=0.7, label="audio")
        ax.scatter(t_xyz[:, 0], t_xyz[:, 1], t_xyz[:, 2], s=6, c="#ef4444", alpha=0.7, label="text")
        cur_gap = gap_distance(a_s, t_s)
        ax.set_title(f"λ={lam:+.2f} → ‖Δ‖={cur_gap:.3f}")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if i == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3d — Optimize the gap under InfoNCE: gradient descent on shift λ
# ---------------------------------------------------------------------------
def optimize_gap(
    audio: np.ndarray,
    text: np.ndarray,
    temperatures: tuple[float, ...],
    n_iters: int = 300,
    lr: float = 0.05,
) -> dict:
    """Treat λ as the only free parameter; gradient-descent it under InfoNCE.

    For each τ, start from λ=0 (no shift) and step λ to minimize the loss. This
    mimics what contrastive optimization does to the gap (without retraining the
    full encoder). The trajectory of the resulting gap distance illustrates the
    paper's claim: low τ keeps a non-zero gap; high τ collapses it.
    """
    audio = l2_normalize(audio); text = l2_normalize(text)
    delta = gap_vector(audio, text, normalize=False)

    out: dict[float, list[dict]] = {}
    for tau in temperatures:
        lam = 0.0
        traj = []
        for _ in range(n_iters):
            # Numerical gradient: central differences (no autograd needed).
            eps = 1e-2
            a_p, t_p = shift_features(audio, text, lam + eps, delta=delta)
            a_m, t_m = shift_features(audio, text, lam - eps, delta=delta)
            l_p = info_nce_loss(a_p, t_p, temperature=tau)
            l_m = info_nce_loss(a_m, t_m, temperature=tau)
            grad = (l_p - l_m) / (2 * eps)
            lam = lam - lr * grad
            a_s, t_s = shift_features(audio, text, lam, delta=delta)
            traj.append(
                {
                    "lambda": float(lam),
                    "gap": float(gap_distance(a_s, t_s, normalize=False)),
                    "loss": float(info_nce_loss(a_s, t_s, temperature=tau)),
                }
            )
        out[float(tau)] = traj
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_loss_landscape(landscape: dict, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(landscape)))
    for (tau, data), color in zip(sorted(landscape.items(), key=lambda x: x[0]), cmap):
        gaps = np.array(data["gaps"])
        losses = np.array(data["losses"])
        # Find where loss is minimized
        idx_min = int(np.argmin(losses))
        ax.plot(gaps, losses, lw=2, color=color, label=f"τ={tau:g}")
        ax.scatter(gaps[idx_min], losses[idx_min], s=80, color=color, zorder=5,
                   edgecolor="black", linewidth=1.0)
    ax.set_xlabel("modality gap distance ‖Δ‖₂")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_optimization(traj_by_tau: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(traj_by_tau)))
    for (tau, traj), color in zip(sorted(traj_by_tau.items()), cmap):
        gaps = [t["gap"] for t in traj]
        losses = [t["loss"] for t in traj]
        axes[0].plot(gaps, lw=2, color=color, label=f"τ={tau:g}")
        axes[1].plot(losses, lw=2, color=color, label=f"τ={tau:g}")
    axes[0].set_xlabel("optimization step"); axes[0].set_ylabel("gap distance ‖Δ‖₂")
    axes[0].set_title("Gap distance trajectory")
    axes[1].set_xlabel("optimization step"); axes[1].set_ylabel("InfoNCE loss")
    axes[1].set_title("Loss trajectory")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_real(backbone: str, split: str = "val"):
    cache = ROOT / "embeddings" / backbone / f"audiocaps__{split}.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"No cache: {cache}.  Run scripts/01_extract_embeddings.py first."
        )
    data = np.load(cache, allow_pickle=True)
    return data["audio"], data["text"], cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["laion", "msclap"], default="laion")
    p.add_argument("--split", default="val")
    p.add_argument("--real", action="store_true",
                   help="Use real CLAP embeddings instead of synthetic.")
    p.add_argument("--allow-synthetic-fallback", action="store_true",
                   help="If set together with --real, missing caches fall back to synthetic.")
    p.add_argument("--n-synth", type=int, default=400)
    p.add_argument("--gap-synth", type=float, default=0.82)
    p.add_argument(
        "--temperatures", nargs="+", type=float,
        default=[1 / 100, 1 / 50, 1 / 30, 1 / 20, 1 / 10, 1.0],
    )
    p.add_argument("--n-lambdas", type=int, default=41)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument(
        "--lambdas-3c", nargs="+", type=float,
        default=[-0.25, 0.0, 0.25, 0.5],
        help="Which λ values to render as 3D-sphere snapshots (3c). Pass more for "
             "the Appendix_3d_sphere.ipynb equivalent.",
    )
    args = p.parse_args()

    if args.real:
        try:
            audio, text, cache = load_real(args.backbone, args.split)
            tag = f"{args.backbone}-audiocaps-{args.split}"
            mode = "real"
            print(f"[real] using {args.backbone} on AudioCaps {args.split} ({audio.shape[0]} pairs)")
        except FileNotFoundError as e:
            if not args.allow_synthetic_fallback:
                print(f"[error] {e}")
                raise SystemExit(1)
            print(f"[warn] {e}\n[warn] falling back to synthetic")
            args.real = False
    if not args.real:
        audio, text = synth_paired_embeddings(
            n=args.n_synth, d=512, gap=args.gap_synth, seed=0,
        )
        tag = f"synth-gap{args.gap_synth}"
        cache = ROOT / "embeddings" / args.backbone / f"audiocaps__{args.split}.npz"
        mode = "synthetic"
        print(f"[synthetic] n={args.n_synth} pairs, target gap={args.gap_synth}")

    out_dir = ROOT / "results" / "figure_3" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": mode,
        "backbone": args.backbone,
        "timestamp": utc_timestamp(),
        "command": command_string(),
        "git_commit": get_git_commit(ROOT),
        "machine_info": machine_info(),
        "input_cache_paths": {"audiocaps_val": file_info(cache)},
        "input_cache_exists": {"audiocaps_val": cache.exists()},
        "data_shapes": {"audio": list(audio.shape), "text": list(text.shape)},
    }

    # 3a
    print("\n[3a] gap stats:")
    stats = gap_stats_on_data(audio, text)
    print(json.dumps(stats, indent=2))
    (out_dir / "3a_gap_stats.json").write_text(json.dumps({**metadata, "results": stats}, indent=2))

    # 3b
    print("\n[3b] sweeping loss landscape...")
    lambdas = np.linspace(-0.5, 1.5, args.n_lambdas)
    landscape = loss_landscape_sweep(
        audio, text, tuple(args.temperatures), lambdas,
    )
    for tau, data in sorted(landscape.items()):
        gaps = np.array(data["gaps"]); losses = np.array(data["losses"])
        idx = int(np.argmin(losses))
        print(f"  τ={tau:.4g}  argmin gap={gaps[idx]:.3f}  loss={losses[idx]:.3f}")
    plot_loss_landscape(
        landscape, out_dir / "3b_loss_landscape.png",
        title=f"Figure 3b — InfoNCE vs gap distance ({tag})",
    )
    (out_dir / "3b_landscape.json").write_text(json.dumps({**metadata, "results": landscape}, indent=2))

    # 3c — 3D sphere snapshots at a few λ values
    print(f"\n[3c] 3D-sphere snapshots at λ={args.lambdas_3c}")
    sphere_viz(
        audio, text,
        lambdas=tuple(args.lambdas_3c),
        out_path=out_dir / "3c_3d_sphere.png",
    )

    # 3d
    print("\n[3d] optimizing λ under InfoNCE...")
    traj = optimize_gap(audio, text, tuple(args.temperatures), n_iters=args.n_iters)
    plot_optimization(traj, out_dir / "3d_optimization.png")
    (out_dir / "3d_optimization.json").write_text(json.dumps({**metadata, "results": traj}, indent=2))
    for tau, t in sorted(traj.items()):
        print(f"  τ={tau:.4g}  final gap={t[-1]['gap']:.3f}  final loss={t[-1]['loss']:.3f}")

    print(f"\nresults under {out_dir}")


if __name__ == "__main__":
    main()
