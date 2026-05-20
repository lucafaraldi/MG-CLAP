#!/usr/bin/env python3
"""MG-CLAP-lite continual learning on cached ESC-50 CLAP embeddings.

This experiment keeps the CLAP backbone frozen and trains a lightweight
audio-side low-rank residual adapter in a class-incremental setting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.gap_utils import gap_distance, l2_normalize  # noqa: E402
from lib.provenance import command_string, file_info, get_git_commit, machine_info, torch_info, utc_timestamp  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["laion", "msclap"], required=True)
    p.add_argument("--fold", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--classes-per-task", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--adapter-rank", type=int, default=16)
    p.add_argument("--adapter-scale", type=float, default=1.0)
    p.add_argument("--epochs-max", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--betas", nargs="+", type=float, default=[0.0, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--out-dir", default="results/continual_mgclap")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--class-order", default="canonical", choices=["canonical", "shuffled"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def required_cache_paths(backbone: str) -> tuple[Path, Path]:
    return (
        ROOT / "embeddings" / backbone / "esc50.npz",
        ROOT / "embeddings" / backbone / "audiocaps__val.npz",
    )


def print_missing_cache_help(backbone: str) -> None:
    print("Missing required caches for real MG-CLAP-lite continual learning.")
    print("Run these commands first:")
    print(
        f"python scripts/01_extract_embeddings.py --backbone {backbone} --dataset audiocaps --split val"
    )
    print(f"python scripts/01_extract_embeddings.py --backbone {backbone} --dataset esc50")


@dataclass
class ESC50Cache:
    audio: np.ndarray
    text: np.ndarray
    labels: np.ndarray
    folds: np.ndarray
    class_names: list[str]
    prompts: list[str]
    cache_path: Path
    mtime: str | None


def load_esc50_cache(backbone: str) -> ESC50Cache:
    esc50_cache, audiocaps_cache = required_cache_paths(backbone)
    if not esc50_cache.exists() or not audiocaps_cache.exists():
        print_missing_cache_help(backbone)
        missing = []
        if not audiocaps_cache.exists():
            missing.append(str(audiocaps_cache))
        if not esc50_cache.exists():
            missing.append(str(esc50_cache))
        raise FileNotFoundError("Missing cache files:\n" + "\n".join(missing))

    data = np.load(esc50_cache, allow_pickle=True)
    meta_raw = data["meta"].item() if getattr(data["meta"], "shape", ()) == () else data["meta"]
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    return ESC50Cache(
        audio=np.asarray(data["audio"], dtype=np.float32),
        text=np.asarray(data["text"], dtype=np.float32),
        labels=np.asarray(data["labels"], dtype=np.int64),
        folds=np.asarray(data["folds"], dtype=np.int64),
        class_names=list(meta["class_names"]),
        prompts=list(meta["prompts"]),
        cache_path=esc50_cache,
        mtime=file_info(esc50_cache)["mtime"],
    )


def build_class_order(n_classes: int, kind: str, seed: int) -> list[int]:
    order = list(range(n_classes))
    if kind == "shuffled":
        rng = np.random.default_rng(seed)
        order = [int(x) for x in rng.permutation(order)]
    return [int(x) for x in order]


def split_tasks(class_order: list[int], tasks: int, classes_per_task: int) -> list[list[int]]:
    if len(class_order) != tasks * classes_per_task:
        raise ValueError(
            f"class partition mismatch: {len(class_order)} classes but tasks*classes_per_task="
            f"{tasks * classes_per_task}"
        )
    return [
        class_order[i * classes_per_task : (i + 1) * classes_per_task]
        for i in range(tasks)
    ]


def normalize_np(x: np.ndarray) -> np.ndarray:
    return l2_normalize(np.asarray(x, dtype=np.float64)).astype(np.float32)


class LowRankResidualAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, scale: float):
        super().__init__()
        self.scale = scale
        self.a = nn.Linear(dim, rank, bias=False)
        self.b = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.a.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.scale * self.b(self.a(x))
        return F.normalize(out, dim=-1)


def clone_adapter(adapter: LowRankResidualAdapter, device: torch.device) -> LowRankResidualAdapter:
    new = LowRankResidualAdapter(adapter.a.in_features, adapter.a.out_features, adapter.scale).to(device)
    new.load_state_dict(adapter.state_dict())
    return new


def adapt_np(adapter: LowRankResidualAdapter | None, x: np.ndarray, device: torch.device) -> np.ndarray:
    if adapter is None:
        return normalize_np(x)
    with torch.no_grad():
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        z = adapter(xt).cpu().numpy()
    return z.astype(np.float32)


def mean_pos_neg(audio_z: np.ndarray, text_seen: np.ndarray, labels_seen_idx: np.ndarray) -> tuple[float, float]:
    sims = audio_z @ text_seen.T
    pos = sims[np.arange(sims.shape[0]), labels_seen_idx]
    if sims.shape[1] <= 1:
        neg_mean = 0.0
    else:
        mask = np.ones_like(sims, dtype=bool)
        mask[np.arange(sims.shape[0]), labels_seen_idx] = False
        neg_mean = float(sims[mask].mean())
    return float(pos.mean()), neg_mean


def compute_gap(audio_z: np.ndarray, text_seen: np.ndarray) -> float:
    return float(gap_distance(audio_z, text_seen, normalize=True))


def select_indices(labels: np.ndarray, allowed_classes: Iterable[int]) -> np.ndarray:
    allowed = np.array(list(allowed_classes), dtype=np.int64)
    return np.isin(labels, allowed)


def map_labels(labels: np.ndarray, seen_classes: list[int]) -> np.ndarray:
    mapping = {c: i for i, c in enumerate(seen_classes)}
    return np.asarray([mapping[int(y)] for y in labels], dtype=np.int64)


def logits_with_optional_proto(
    audio_z: np.ndarray,
    text_seen: np.ndarray,
    prototype_matrix: np.ndarray | None,
    beta: float,
    temperature: float,
) -> np.ndarray:
    logits = (audio_z @ text_seen.T) / temperature
    if prototype_matrix is not None:
        logits = logits + beta * ((audio_z @ prototype_matrix.T) / temperature)
    return logits


def evaluate_seen(
    adapter: LowRankResidualAdapter | None,
    audio_eval: np.ndarray,
    labels_eval: np.ndarray,
    seen_classes: list[int],
    text_full: np.ndarray,
    prototypes: dict[int, np.ndarray],
    beta: float,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    text_seen = normalize_np(text_full[seen_classes])
    label_idx = map_labels(labels_eval, seen_classes)
    audio_z = adapt_np(adapter, audio_eval, device)
    proto_matrix = None
    if prototypes:
        proto_matrix = normalize_np(np.stack([prototypes[c] for c in seen_classes], axis=0))
    logits = logits_with_optional_proto(audio_z, text_seen, proto_matrix, beta, temperature)
    pred = logits.argmax(axis=1)
    acc = float((pred == label_idx).mean()) if len(label_idx) else 0.0
    pos_mean, neg_mean = mean_pos_neg(audio_z, text_seen, label_idx)
    return {
        "accuracy": acc,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "gap_distance": compute_gap(audio_z, text_seen),
    }


def per_task_eval_accuracies(
    adapter: LowRankResidualAdapter | None,
    eval_audio: np.ndarray,
    eval_labels: np.ndarray,
    text_full: np.ndarray,
    tasks: list[list[int]],
    seen_upto: int,
    prototypes: dict[int, np.ndarray],
    beta: float,
    temperature: float,
    device: torch.device,
) -> dict[int, float]:
    out: dict[int, float] = {}
    seen_classes = [c for task in tasks[: seen_upto + 1] for c in task]
    text_seen = normalize_np(text_full[seen_classes])
    proto_matrix = None
    if prototypes:
        proto_matrix = normalize_np(np.stack([prototypes[c] for c in seen_classes], axis=0))
    for task_id in range(seen_upto + 1):
        task_classes = tasks[task_id]
        mask = select_indices(eval_labels, task_classes)
        if not np.any(mask):
            out[task_id] = float("nan")
            continue
        audio_z = adapt_np(adapter, eval_audio[mask], device)
        label_idx = map_labels(eval_labels[mask], seen_classes)
        logits = logits_with_optional_proto(audio_z, text_seen, proto_matrix, beta, temperature)
        pred = logits.argmax(axis=1)
        out[task_id] = float((pred == label_idx).mean())
    return out


def train_task(
    adapter: LowRankResidualAdapter,
    audio_train: np.ndarray,
    labels_train: np.ndarray,
    seen_classes: list[int],
    text_full: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    device: torch.device,
) -> None:
    adapter.train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.from_numpy(np.asarray(audio_train, dtype=np.float32)).to(device)
    y = torch.from_numpy(map_labels(labels_train, seen_classes)).to(device)
    text_seen = torch.from_numpy(normalize_np(text_full[seen_classes])).to(device)

    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]
            z = adapter(xb)
            logits = (z @ text_seen.T) / temperature
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def preservation_probe(
    base_adapter: LowRankResidualAdapter,
    audio_train: np.ndarray,
    labels_train: np.ndarray,
    task_classes: list[int],
    text_full: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
) -> tuple[int, list[dict[str, float | int | bool]], LowRankResidualAdapter]:
    probe = clone_adapter(base_adapter, device)
    seen_text = normalize_np(text_full[task_classes])
    labels_idx = map_labels(labels_train, task_classes)

    def metrics() -> tuple[float, float, float]:
        audio_z = adapt_np(probe, audio_train, device)
        pos_mean, neg_mean = mean_pos_neg(audio_z, seen_text, labels_idx)
        gap = compute_gap(audio_z, seen_text)
        return pos_mean, neg_mean, gap

    pos0, neg0, gap0 = metrics()
    history: list[dict[str, float | int | bool]] = [
        {
            "fold": fold,
            "seed": args.seed,
            "epoch": 0,
            "neg_mean": neg0,
            "pos_mean": pos0,
            "drift": 0.0,
            "gap_distance": gap0,
            "selected": False,
        }
    ]

    checkpoints: dict[int, dict[str, torch.Tensor]] = {0: {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}}
    selected_epoch = args.epochs_max
    crossed = False
    for epoch in range(1, args.epochs_max + 1):
        train_task(
            probe,
            audio_train,
            labels_train,
            task_classes,
            text_full,
            epochs=1,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            device=device,
        )
        pos_e, neg_e, gap_e = metrics()
        drift = abs(neg_e - neg0) / max(abs(neg0), 1e-8)
        history.append(
            {
                "fold": fold,
                "seed": args.seed,
                "epoch": epoch,
                "neg_mean": neg_e,
                "pos_mean": pos_e,
                "drift": drift,
                "gap_distance": gap_e,
                "selected": False,
            }
        )
        checkpoints[epoch] = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        if not crossed and drift > args.alpha:
            selected_epoch = 1 if epoch == 1 else epoch - 1
            crossed = True
            break
    if not crossed:
        selected_epoch = args.epochs_max
    for row in history:
        if row["epoch"] == selected_epoch:
            row["selected"] = True

    selected_adapter = clone_adapter(base_adapter, device)
    selected_adapter.load_state_dict(checkpoints[selected_epoch])
    return selected_epoch, history, selected_adapter


def update_prototypes(
    adapter: LowRankResidualAdapter | None,
    audio_train: np.ndarray,
    labels_train: np.ndarray,
    classes_current: list[int],
    prototypes: dict[int, np.ndarray],
    device: torch.device,
) -> None:
    for c in classes_current:
        mask = labels_train == c
        if not np.any(mask):
            continue
        z = adapt_np(adapter, audio_train[mask], device)
        proto = normalize_np(z.mean(axis=0, keepdims=True))[0]
        prototypes[int(c)] = proto.astype(np.float32)


def first_last_forgetting(per_task_accs: dict[int, list[float]]) -> float:
    drops = []
    for task_id, vals in per_task_accs.items():
        clean = [v for v in vals if not math.isnan(v)]
        if len(clean) >= 2:
            drops.append(clean[0] - clean[-1])
    return float(np.mean(drops)) if drops else 0.0


def method_beta_grid(method: str, betas: list[float]) -> list[float]:
    return betas if method in {"mgc_only", "mgp_mgc"} else [0.0]


def run_zero_shot(
    fold: int,
    esc50: ESC50Cache,
    task_splits: list[list[int]],
    eval_audio: np.ndarray,
    eval_labels: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[int, list[float]]]:
    curves = []
    task_accs: dict[int, list[float]] = {}
    for task_id in range(len(task_splits)):
        seen_classes = [c for task in task_splits[: task_id + 1] for c in task]
        mask = select_indices(eval_labels, seen_classes)
        metrics = evaluate_seen(
            None,
            eval_audio[mask],
            eval_labels[mask],
            seen_classes,
            esc50.text,
            {},
            beta=0.0,
            temperature=args.temperature,
            device=device,
        )
        curves.append(
            {
                "method": "zero_shot_clap",
                "beta": 0.0,
                "fold": fold,
                "seed": args.seed,
                "task_id": task_id + 1,
                "seen_classes": len(seen_classes),
                **metrics,
            }
        )
        eval_by_task = per_task_eval_accuracies(
            None,
            eval_audio,
            eval_labels,
            esc50.text,
            task_splits,
            task_id,
            {},
            beta=0.0,
            temperature=args.temperature,
            device=device,
        )
        for tid, acc in eval_by_task.items():
            task_accs.setdefault(tid, []).append(acc)
    return curves, task_accs


def run_sequential_method(
    method: str,
    fold: int,
    esc50: ESC50Cache,
    task_splits: list[list[int]],
    train_audio: np.ndarray,
    train_labels: np.ndarray,
    eval_audio: np.ndarray,
    eval_labels: np.ndarray,
    e_star: int | None,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[float, dict[int, list[float]]]]:
    base_adapter = LowRankResidualAdapter(esc50.audio.shape[1], args.adapter_rank, args.adapter_scale).to(device)
    beta_curves: list[dict] = []
    beta_task_acc_store: dict[float, dict[int, list[float]]] = {}

    use_preservation = method in {"mgp_only", "mgp_mgc"}
    use_prototypes = method in {"mgc_only", "mgp_mgc"}

    for beta in method_beta_grid(method, args.betas):
        adapter = clone_adapter(base_adapter, device)
        prototypes: dict[int, np.ndarray] = {}
        task_acc_lists: dict[int, list[float]] = {}
        if use_preservation:
            first_classes = task_splits[0]
            first_mask = select_indices(train_labels, first_classes)
            _, _, selected_adapter = preservation_probe(
                adapter,
                train_audio[first_mask],
                train_labels[first_mask],
                first_classes,
                esc50.text,
                args,
                device,
                fold,
            )
            adapter = selected_adapter
            if use_prototypes:
                update_prototypes(
                    adapter,
                    train_audio[first_mask],
                    train_labels[first_mask],
                    first_classes,
                    prototypes,
                    device,
                )
            seen_classes = [c for task in task_splits[:1] for c in task]
            eval_mask = select_indices(eval_labels, seen_classes)
            metrics = evaluate_seen(
                adapter,
                eval_audio[eval_mask],
                eval_labels[eval_mask],
                seen_classes,
                esc50.text,
                prototypes,
                beta=beta,
                temperature=args.temperature,
                device=device,
            )
            beta_curves.append(
                {
                    "method": method,
                    "beta": beta,
                    "fold": fold,
                    "seed": args.seed,
                    "task_id": 1,
                    "seen_classes": len(seen_classes),
                    **metrics,
                }
            )
            first_eval = per_task_eval_accuracies(
                adapter,
                eval_audio,
                eval_labels,
                esc50.text,
                task_splits,
                0,
                prototypes,
                beta=beta,
                temperature=args.temperature,
                device=device,
            )
            for tid, acc in first_eval.items():
                task_acc_lists.setdefault(tid, []).append(acc)
            start_task = 1
        else:
            start_task = 0

        epochs_per_task = e_star if use_preservation else args.epochs_max
        for task_id in range(start_task, len(task_splits)):
            current_classes = task_splits[task_id]
            seen_classes = [c for task in task_splits[: task_id + 1] for c in task]
            current_mask = select_indices(train_labels, current_classes)
            train_task(
                adapter,
                train_audio[current_mask],
                train_labels[current_mask],
                seen_classes,
                esc50.text,
                epochs=epochs_per_task,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                device=device,
            )
            if use_prototypes:
                update_prototypes(
                    adapter,
                    train_audio[current_mask],
                    train_labels[current_mask],
                    current_classes,
                    prototypes,
                    device,
                )
            eval_mask = select_indices(eval_labels, seen_classes)
            metrics = evaluate_seen(
                adapter,
                eval_audio[eval_mask],
                eval_labels[eval_mask],
                seen_classes,
                esc50.text,
                prototypes,
                beta=beta,
                temperature=args.temperature,
                device=device,
            )
            beta_curves.append(
                {
                    "method": method,
                    "beta": beta,
                    "fold": fold,
                    "seed": args.seed,
                    "task_id": task_id + 1,
                    "seen_classes": len(seen_classes),
                    **metrics,
                }
            )
            eval_by_task = per_task_eval_accuracies(
                adapter,
                eval_audio,
                eval_labels,
                esc50.text,
                task_splits,
                task_id,
                prototypes,
                beta=beta,
                temperature=args.temperature,
                device=device,
            )
            for tid, acc in eval_by_task.items():
                task_acc_lists.setdefault(tid, []).append(acc)

        beta_task_acc_store[beta] = task_acc_lists
        if args.debug:
            print(f"[debug] {method} fold={fold} beta={beta} complete")

    return beta_curves, beta_task_acc_store


def final_rows_from_curves(
    curves: list[dict],
    e_star_by_fold: dict[int, int | None],
    forgetting_lookup: dict[tuple[str, float, int, int], dict[int, list[float]]],
) -> list[dict]:
    rows = []
    grouped: dict[tuple[str, float, int, int], list[dict]] = {}
    for row in curves:
        key = (row["method"], float(row["beta"]), int(row["fold"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    for (method, beta, fold, seed), seq in sorted(grouped.items()):
        seq = sorted(seq, key=lambda r: r["task_id"])
        avg = float(np.mean([r["accuracy"] for r in seq]))
        last = float(seq[-1]["accuracy"])
        task_histories = forgetting_lookup.get((method, beta, fold, seed), {})
        rows.append(
            {
                "method": method,
                "beta": beta,
                "fold": fold,
                "seed": seed,
                "Avg": avg,
                "Last": last,
                "forgetting_proxy": first_last_forgetting(task_histories),
                "e_star": e_star_by_fold.get(fold),
            }
        )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], float(row["beta"])), []).append(row)
    out = []
    for (method, beta), vals in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "beta": beta,
                "folds": len(vals),
                "Avg_mean": float(np.mean([v["Avg"] for v in vals])),
                "Avg_std": float(np.std([v["Avg"] for v in vals])),
                "Last_mean": float(np.mean([v["Last"] for v in vals])),
                "Last_std": float(np.std([v["Last"] for v in vals])),
                "forgetting_proxy_mean": float(np.mean([v["forgetting_proxy"] for v in vals])),
                "forgetting_proxy_std": float(np.std([v["forgetting_proxy"] for v in vals])),
            }
        )
    return out


def choose_best_betas(rows: list[dict], method: str) -> tuple[float | None, float | None]:
    candidates = [r for r in rows if r["method"] == method]
    if not candidates:
        return None, None
    by_beta: dict[float, list[dict]] = {}
    for row in candidates:
        by_beta.setdefault(float(row["beta"]), []).append(row)
    avg_best = max(by_beta.items(), key=lambda kv: np.mean([r["Avg"] for r in kv[1]]))[0]
    last_best = max(by_beta.items(), key=lambda kv: np.mean([r["Last"] for r in kv[1]]))[0]
    return avg_best, last_best


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_accuracy_curve(curves: list[dict], final_rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["zero_shot_clap", "naive_adapter", "mgp_only", "mgc_only", "mgp_mgc"]
    best_beta = {
        "mgc_only": choose_best_betas(final_rows, "mgc_only")[0],
        "mgp_mgc": choose_best_betas(final_rows, "mgp_mgc")[0],
    }
    palette = {
        "zero_shot_clap": "#6b7280",
        "naive_adapter": "#ef4444",
        "mgp_only": "#2563eb",
        "mgc_only": "#10b981",
        "mgp_mgc": "#7c3aed",
    }
    for method in methods:
        rows = [r for r in curves if r["method"] == method]
        if method in best_beta:
            rows = [r for r in rows if float(r["beta"]) == float(best_beta[method])]
        else:
            rows = [r for r in rows if float(r["beta"]) == 0.0]
        if not rows:
            continue
        task_ids = sorted({int(r["task_id"]) for r in rows})
        means = []
        for tid in task_ids:
            vals = [float(r["accuracy"]) for r in rows if int(r["task_id"]) == tid]
            means.append(float(np.mean(vals)))
        label = method if method not in best_beta else f"{method} (best β={best_beta[method]:g})"
        ax.plot(task_ids, means, marker="o", lw=2, label=label, color=palette[method])
    ax.set_xlabel("task")
    ax.set_ylabel("accuracy on seen classes")
    ax.set_title("MG-CLAP-lite continual accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap_trajectory(curves: list[dict], final_rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = ["naive_adapter", "mgp_only", "mgp_mgc"]
    best_beta = {"mgp_mgc": choose_best_betas(final_rows, "mgp_mgc")[0]}
    palette = {"naive_adapter": "#ef4444", "mgp_only": "#2563eb", "mgp_mgc": "#7c3aed"}
    for method in methods:
        rows = [r for r in curves if r["method"] == method]
        if method in best_beta:
            rows = [r for r in rows if float(r["beta"]) == float(best_beta[method])]
        else:
            rows = [r for r in rows if float(r["beta"]) == 0.0]
        if not rows:
            continue
        task_ids = sorted({int(r["task_id"]) for r in rows})
        neg = [float(np.mean([r["neg_mean"] for r in rows if int(r["task_id"]) == tid])) for tid in task_ids]
        gap = [float(np.mean([r["gap_distance"] for r in rows if int(r["task_id"]) == tid])) for tid in task_ids]
        axes[0].plot(task_ids, neg, marker="o", lw=2, label=method, color=palette[method])
        axes[1].plot(task_ids, gap, marker="o", lw=2, label=method, color=palette[method])
    axes[0].set_title("Negative audio-text similarity")
    axes[0].set_xlabel("task")
    axes[0].set_ylabel("neg mean")
    axes[1].set_title("Gap distance trajectory")
    axes[1].set_xlabel("task")
    axes[1].set_ylabel("gap distance")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_beta_sweep(final_rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, color in [("mgc_only", "#10b981"), ("mgp_mgc", "#7c3aed")]:
        rows = [r for r in final_rows if r["method"] == method]
        if not rows:
            continue
        betas = sorted({float(r["beta"]) for r in rows})
        avg = [float(np.mean([r["Avg"] for r in rows if float(r["beta"]) == b])) for b in betas]
        last = [float(np.mean([r["Last"] for r in rows if float(r["beta"]) == b])) for b in betas]
        axes[0].plot(betas, avg, marker="o", lw=2, label=method, color=color)
        axes[1].plot(betas, last, marker="o", lw=2, label=method, color=color)
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("Avg")
    axes[0].set_title("Beta sweep by Avg")
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("Last")
    axes[1].set_title("Beta sweep by Last")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_run_report(
    args: argparse.Namespace,
    summary: dict,
    final_rows: list[dict],
) -> str:
    lines = [
        "# MG-CLAP-lite Run Report",
        "",
        "## What Was Run",
        "",
        f"- Command: `{summary['command']}`",
        f"- Backbone: `{args.backbone}`",
        f"- Folds: `{summary['fold_list']}`",
        f"- Tasks x classes/task: `{args.tasks} x {args.classes_per_task}`",
        f"- Dry run: `{args.dry_run}`",
        "",
        "## What Was Implemented",
        "",
        "- Zero-shot CLAP baseline",
        "- Sequential low-rank residual audio adapter",
        "- Modality-gap preservation epoch selection via task-1 negative-similarity drift",
        "- Audio prototype compensation head with explicit beta sweep",
        "- Fold-wise summary, trajectories, and provenance metadata",
        "",
        "## What Remains Synthetic Elsewhere In The Repo",
        "",
        "- Current saved Figure 2 outputs are still synthetic/random-init stand-ins.",
        "- Current saved Figure 3 outputs may still be synthetic unless re-run with `--real`.",
        "- Current saved Table 1 shift outputs may still be synthetic unless re-run in shift mode with real caches.",
        "",
        "## Main Table",
        "",
        "| method | beta | fold | Avg | Last | forgetting_proxy | e_star |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['method']} | {row['beta']} | {row['fold']} | {row['Avg']:.4f} | "
            f"{row['Last']:.4f} | {row['forgetting_proxy']:.4f} | {row['e_star']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `mgp_only` tests whether limiting drift in negative audio-text similarity stabilizes continual learning.",
            "- `mgc_only` tests whether audio-space prototypes compensate for text-only classifier limits.",
            "- `mgp_mgc` tests the combination. Beta is an exploratory sweep, not a hidden validation-tuned parameter.",
            "",
            "## Limitations",
            "",
            "- Frozen-embedding MG-CLAP-lite is not full CLAP fine-tuning.",
            "- No replay buffer is used.",
            "- Beta sweep is exploratory and uses the evaluation folds directly in the current implementation summary.",
            "- This run does not retroactively certify provenance of older result files elsewhere in the repo.",
            "",
            "## Presentation-Safe Claims",
            "",
        ]
    )
    lines.extend([f"- {claim}" for claim in summary["presentation_safe_claims"]])
    lines.extend(
        [
            "",
            "## Claims To Avoid",
            "",
        ]
    )
    lines.extend([f"- {claim}" for claim in summary["dangerous_claims_to_avoid"]])
    lines.extend(
        [
            "",
            "## Next Experiments",
            "",
            "- Run all five ESC-50 folds for both backbones.",
            "- Compare best-beta-by-Avg against the text-only continual baselines.",
            "- Re-run real Table 1 shift and real Figure 3 with provenance-enabled summaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    try:
        esc50 = load_esc50_cache(args.backbone)
    except FileNotFoundError as e:
        print(str(e))
        raise SystemExit(1)
    n_classes = esc50.text.shape[0]
    class_order = build_class_order(n_classes, args.class_order, args.seed)
    task_splits = split_tasks(class_order, args.tasks, args.classes_per_task)

    fold_list = [1, 2, 3, 4, 5] if args.fold == "all" else [int(args.fold)]
    if any(f not in (1, 2, 3, 4, 5) for f in fold_list):
        raise ValueError("--fold must be 'all' or an integer in 1..5")

    out_root = ROOT / args.out_dir / args.backbone
    if args.dry_run:
        out_root = out_root / "dry_run"
        fold_list = fold_list[:1]
        task_splits = task_splits[:2]
        args.epochs_max = min(args.epochs_max, 2)

    out_root.mkdir(parents=True, exist_ok=True)

    train_eval_curves: list[dict] = []
    gap_rows: list[dict] = []
    preservation_rows: list[dict] = []
    e_star_by_fold: dict[int, int | None] = {}
    forgetting_lookup: dict[tuple[str, float, int, int], dict[int, list[float]]] = {}

    methods_run = ["zero_shot_clap", "naive_adapter", "mgp_only", "mgc_only", "mgp_mgc"]

    for fold in fold_list:
        train_mask = esc50.folds != fold
        eval_mask = esc50.folds == fold
        train_audio = esc50.audio[train_mask]
        train_labels = esc50.labels[train_mask]
        eval_audio = esc50.audio[eval_mask]
        eval_labels = esc50.labels[eval_mask]

        first_task_classes = task_splits[0]
        first_mask = select_indices(train_labels, first_task_classes)
        probe_base = LowRankResidualAdapter(esc50.audio.shape[1], args.adapter_rank, args.adapter_scale).to(device)
        e_star, probe_hist, _ = preservation_probe(
            probe_base,
            train_audio[first_mask],
            train_labels[first_mask],
            first_task_classes,
            esc50.text,
            args,
            device,
            fold,
        )
        e_star_by_fold[fold] = e_star
        preservation_rows.extend(probe_hist)

        zero_curves, zero_task_accs = run_zero_shot(
            fold, esc50, task_splits, eval_audio, eval_labels, device, args
        )
        train_eval_curves.extend(zero_curves)
        forgetting_lookup[("zero_shot_clap", 0.0, fold, args.seed)] = zero_task_accs

        for method in ["naive_adapter", "mgp_only", "mgc_only", "mgp_mgc"]:
            curves, beta_task_accs = run_sequential_method(
                method,
                fold,
                esc50,
                task_splits,
                train_audio,
                train_labels,
                eval_audio,
                eval_labels,
                e_star,
                device,
                args,
            )
            train_eval_curves.extend(curves)
            for beta, task_accs in beta_task_accs.items():
                forgetting_lookup[(method, float(beta), fold, args.seed)] = task_accs

    for row in train_eval_curves:
        gap_rows.append(
            {
                "method": row["method"],
                "beta": row["beta"],
                "fold": row["fold"],
                "seed": row["seed"],
                "task_id": row["task_id"],
                "pos_mean": row["pos_mean"],
                "neg_mean": row["neg_mean"],
                "gap_distance": row["gap_distance"],
            }
        )

    final_rows = final_rows_from_curves(train_eval_curves, e_star_by_fold, forgetting_lookup)
    aggregate = aggregate_rows(final_rows)

    summary = {
        "timestamp": utc_timestamp(),
        "git_commit": get_git_commit(ROOT),
        "command": command_string(),
        "args": vars(args),
        "machine_info": machine_info(),
        "torch_info": torch_info(),
        "cache_paths": {
            "esc50": file_info(required_cache_paths(args.backbone)[0]),
            "audiocaps_val": file_info(required_cache_paths(args.backbone)[1]),
        },
        "data_shapes": {
            "esc50_audio": list(esc50.audio.shape),
            "esc50_text": list(esc50.text.shape),
            "esc50_labels": list(esc50.labels.shape),
            "esc50_folds": list(esc50.folds.shape),
        },
        "class_order": class_order,
        "class_names": esc50.class_names,
        "fold_list": fold_list,
        "methods_run": methods_run,
        "e_star_per_fold": e_star_by_fold,
        "final_table": final_rows,
        "aggregate_mean_std": aggregate,
        "presentation_safe_claims": [
            "MG-CLAP-lite runs real ESC-50 continual-learning experiments only when real CLAP caches are present.",
            "The adapter starts as an identity-like residual map and does not silently fall back to synthetic data.",
            "The preservation rule is explicitly defined by task-1 negative audio-text similarity drift and recorded per fold.",
            "Prototype compensation is reported as an exploratory beta sweep, not as silently tuned test-time optimization.",
        ],
        "dangerous_claims_to_avoid": [
            "Do not claim this is full CLAP continual fine-tuning; it is a frozen-embedding adapter study.",
            "Do not claim beta was tuned on a held-out validation split in the current exploratory sweep.",
            "Do not claim synthetic Figure 2 or Figure 3 artifacts elsewhere in the repo are real without re-running them.",
            "Do not claim modality-gap preservation transfers unless the real continual-learning run is actually executed and inspected.",
        ],
        "limitations": [
            "No replay buffer or rehearsal memory is used.",
            "Only the audio side is adapted; text embeddings remain frozen.",
            "Continual results depend on cached ESC-50 and AudioCaps embeddings already extracted for the chosen backbone.",
            "Best-beta summaries are exploratory because they summarize evaluation-fold performance.",
        ],
    }

    write_csv(
        out_root / "accuracy_table.csv",
        final_rows,
        ["method", "beta", "fold", "seed", "Avg", "Last", "forgetting_proxy", "e_star"],
    )
    write_csv(
        out_root / "task_curve.csv",
        train_eval_curves,
        ["method", "beta", "fold", "seed", "task_id", "seen_classes", "accuracy", "pos_mean", "neg_mean", "gap_distance"],
    )
    write_csv(
        out_root / "preservation_probe.csv",
        preservation_rows,
        ["fold", "seed", "epoch", "neg_mean", "pos_mean", "drift", "gap_distance", "selected"],
    )
    write_csv(
        out_root / "gap_trajectory.csv",
        gap_rows,
        ["method", "beta", "fold", "seed", "task_id", "pos_mean", "neg_mean", "gap_distance"],
    )
    plot_accuracy_curve(train_eval_curves, final_rows, out_root / "accuracy_curve.png")
    plot_gap_trajectory(train_eval_curves, final_rows, out_root / "gap_trajectory.png")
    plot_beta_sweep(final_rows, out_root / "beta_sweep.png")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "RUN_REPORT.md").write_text(build_run_report(args, summary, final_rows))

    print(f"MG-CLAP-lite results written to {out_root}")
    print("If you need to generate caches first, run:")
    print(
        f"  python scripts/01_extract_embeddings.py --backbone {args.backbone} --dataset audiocaps --split val"
    )
    print(f"  python scripts/01_extract_embeddings.py --backbone {args.backbone} --dataset esc50")


if __name__ == "__main__":
    main()
