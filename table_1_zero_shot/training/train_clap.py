#!/usr/bin/env python3
"""Lightweight CLAP contrastive training — reproduces the design of the original
repo's `Table_1.../training/train_clip.py` without the multi-day cost.

What it does
------------
1. Loads PRE-EXTRACTED CLAP embeddings (audio, text) on AudioCaps. Both
   backbones stay frozen — we only train two small linear projection heads on
   top, plus optionally the log-temperature.
2. Plants a controllable INITIAL gap by biasing the audio head's projection to
   +init_gap/2 along the first basis vector and the text head to -init_gap/2.
3. Trains with symmetric InfoNCE for `--epochs` epochs at a chosen fixed
   temperature τ.
4. Records the gap distance over training and the final τ. Reproduces the
   paper's finding: small τ keeps a non-zero gap, large τ closes it; the
   initial gap influences but does not determine the final one.

Run sweeps
----------
The script's intended use is via `--temperatures` / `--init-gaps` sweeps:

    # Sweep τ with zero initial gap
    python table_1_zero_shot/training/train_clap.py \\
        --backbone laion --epochs 30 \\
        --temperatures 0.01 0.02 0.05 0.1 0.5 1.0

    # Sweep initial-gap at τ=0.07
    python table_1_zero_shot/training/train_clap.py \\
        --backbone laion --epochs 30 \\
        --temperatures 0.07 --init-gaps 0.0 0.3 0.6 0.9

The aggregated result is `results/table_1_training/<backbone>/sweep.json` plus
two-panel `sweep_gap_vs_tau.png` / `sweep_gap_vs_init.png` figures.

Synthetic fallback
------------------
If no real CLAP embedding cache is present the script falls back to
`lib.synth.synth_paired_embeddings`, which gives you a sanity-test run end-to-end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from table_1_zero_shot.training.datasets import EmbeddingPairs  # noqa: E402
from table_1_zero_shot.training.utils import (  # noqa: E402
    DualEncoderHead, info_nce_torch, measure_gap,
)
from lib.provenance import command_string, file_info, get_git_commit, machine_info, torch_info, utc_timestamp  # noqa: E402
from lib.synth import synth_paired_embeddings  # noqa: E402


# ---------------------------------------------------------------------------
def load_embeddings(backbone: str, split: str, synthetic_fallback: bool):
    cache = ROOT / "embeddings" / backbone / f"audiocaps__{split}.npz"
    if cache.exists():
        ds = EmbeddingPairs.from_cache(cache)
        print(f"[real] loaded {len(ds)} pairs from {cache}")
        return ds, "real", cache
    if not synthetic_fallback:
        raise FileNotFoundError(f"no cache and synthetic fallback disabled: {cache}")
    print(f"[warn] no cache at {cache}; falling back to synthetic 400 pairs")
    a, t = synth_paired_embeddings(n=400, d=512, gap=0.6, seed=0)
    return EmbeddingPairs(a.astype(np.float32), t.astype(np.float32)), "synthetic", cache


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
def train_one(
    ds: EmbeddingPairs,
    audio_dim: int,
    text_dim: int,
    *,
    init_gap: float,
    fixed_temperature: Optional[float],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    learn_tau = fixed_temperature is None
    model = DualEncoderHead(
        audio_dim=audio_dim, text_dim=text_dim,
        out_dim=min(audio_dim, text_dim),
        init_gap=init_gap,
        learn_temperature=learn_tau,
        init_temperature=fixed_temperature if fixed_temperature else 0.07,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    audio_all = ds.audio.to(device)
    text_all = ds.text.to(device)

    history: list[dict] = []
    # Initial measurement
    init_gap_post = measure_gap(model, audio_all, text_all)
    history.append({"epoch": 0, "gap": init_gap_post, "loss": float("nan"),
                    "tau": float(model.temperature().item())})

    for ep in range(1, epochs + 1):
        # Shuffled mini-batches
        perm = torch.randperm(len(ds), device=device)
        epoch_losses = []
        for i in range(0, len(ds) - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            a_b = audio_all[idx]
            t_b = text_all[idx]
            a_proj, t_proj = model(a_b, t_b)
            tau = model.temperature() if learn_tau else torch.tensor(
                fixed_temperature, device=device
            )
            loss = info_nce_torch(a_proj, t_proj, tau)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_losses.append(float(loss.item()))
        history.append({
            "epoch": ep,
            "gap": measure_gap(model, audio_all, text_all),
            "loss": float(np.mean(epoch_losses)),
            "tau": float(model.temperature().item()),
        })

    return {
        "config": {
            "init_gap": init_gap,
            "fixed_temperature": fixed_temperature,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "audio_dim": audio_dim,
            "text_dim": text_dim,
        },
        "history": history,
        "final_gap": history[-1]["gap"],
        "final_tau": history[-1]["tau"],
        "final_loss": history[-1]["loss"],
    }


def sweep(
    ds: EmbeddingPairs,
    *,
    temperatures: Iterable[float],
    init_gaps: Iterable[float],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seeds: Iterable[int] = (0,),
):
    audio_dim = ds.audio.shape[1]
    text_dim = ds.text.shape[1]
    results = []
    for tau in temperatures:
        for init_gap in init_gaps:
            for s in seeds:
                print(f"  τ={tau:.4g}  init_gap={init_gap:.3f}  seed={s}")
                r = train_one(
                    ds, audio_dim=audio_dim, text_dim=text_dim,
                    init_gap=init_gap, fixed_temperature=tau,
                    epochs=epochs, batch_size=batch_size, lr=lr,
                    device=device, seed=s,
                )
                r["config"]["tau"] = tau
                print(f"    → final gap {r['final_gap']:.3f}  loss {r['final_loss']:.3f}")
                results.append(r)
    return results


# ---------------------------------------------------------------------------
def plot_sweeps(results: list[dict], out_dir: Path):
    # Aggregate by (tau, init_gap) → final_gap (mean over seeds)
    from collections import defaultdict

    by_tau_init = defaultdict(list)
    for r in results:
        c = r["config"]
        by_tau_init[(c["tau"], c["init_gap"])].append(r["final_gap"])

    taus = sorted({k[0] for k in by_tau_init})
    inits = sorted({k[1] for k in by_tau_init})

    # Panel 1: final gap vs τ for each init_gap
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, max(1, len(inits))))
    for init_g, color in zip(inits, cmap):
        ys = [np.mean(by_tau_init[(t, init_g)]) for t in taus]
        es = [np.std(by_tau_init[(t, init_g)]) if len(by_tau_init[(t, init_g)]) > 1 else 0
              for t in taus]
        axes[0].errorbar(taus, ys, yerr=es, lw=2, marker="o", color=color,
                         label=f"init_gap={init_g:.2f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("temperature τ")
    axes[0].set_ylabel("final ‖Δ‖₂")
    axes[0].set_title("Final gap vs τ")
    axes[0].legend(frameon=False, fontsize=8)

    # Panel 2: final gap vs init_gap for each τ
    cmap = plt.cm.plasma(np.linspace(0, 1, max(1, len(taus))))
    for tau, color in zip(taus, cmap):
        ys = [np.mean(by_tau_init[(tau, i)]) for i in inits]
        es = [np.std(by_tau_init[(tau, i)]) if len(by_tau_init[(tau, i)]) > 1 else 0
              for i in inits]
        axes[1].errorbar(inits, ys, yerr=es, lw=2, marker="o", color=color,
                         label=f"τ={tau:.2g}")
    axes[1].plot([0, max(inits)], [0, max(inits)], color="black", lw=0.7,
                 linestyle=":", label="y=x (gap preserved)")
    axes[1].set_xlabel("initial ‖Δ‖₂")
    axes[1].set_ylabel("final ‖Δ‖₂")
    axes[1].set_title("Final gap vs initial gap")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Table 1 — light contrastive training sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_final_gap.png", dpi=150)
    plt.close(fig)

    # Training curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for r, color in zip(results, cmap):
        c = r["config"]
        epochs = [h["epoch"] for h in r["history"]]
        gaps = [h["gap"] for h in r["history"]]
        label = f"τ={c['tau']:.2g} init={c['init_gap']:.2f}"
        ax.plot(epochs, gaps, lw=1.4, color=color, alpha=0.8, label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel("‖Δ‖₂")
    ax.set_title("Gap distance during training")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["laion", "msclap"], default="laion")
    p.add_argument("--split", default="val")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--temperatures", nargs="+", type=float,
        default=[0.01, 0.05, 0.1, 0.5, 1.0],
    )
    p.add_argument(
        "--init-gaps", nargs="+", type=float,
        default=[0.0, 0.3, 0.6, 0.9],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--allow-synthetic", action="store_true",
                   help="Fall back to synthetic data if cache missing.")
    args = p.parse_args()

    out_dir = ROOT / "results" / "table_1_training" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"device: {device}")

    ds, mode, cache_path = load_embeddings(args.backbone, args.split, synthetic_fallback=args.allow_synthetic)
    print(f"dataset: {len(ds)} pairs  audio_dim={ds.audio.shape[1]}  text_dim={ds.text.shape[1]}")

    print(f"\nsweep: |τ|={len(args.temperatures)}  |init|={len(args.init_gaps)}  |seeds|={len(args.seeds)}")
    results = sweep(
        ds,
        temperatures=args.temperatures,
        init_gaps=args.init_gaps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seeds=args.seeds,
    )

    payload = {
        "mode": mode,
        "backbone": args.backbone,
        "timestamp": utc_timestamp(),
        "command": command_string(),
        "git_commit": get_git_commit(ROOT),
        "machine_info": machine_info(),
        "torch_info": torch_info(),
        "input_cache_paths": {"audiocaps_val": file_info(cache_path)},
        "input_cache_exists": {"audiocaps_val": cache_path.exists()},
        "data_shapes": {"audio": list(ds.audio.shape), "text": list(ds.text.shape)},
        "results": results,
    }
    (out_dir / "sweep.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "sweep_legacy_results.json").write_text(json.dumps(results, indent=2))
    plot_sweeps(results, out_dir)
    print(f"\nresults under {out_dir}")


if __name__ == "__main__":
    main()
