#!/usr/bin/env python3
"""Summarize Phase 2 MG-CLAP-lite robustness experiments."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.provenance import command_string, file_info, get_git_commit, machine_info, utc_timestamp  # noqa: E402


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect_phase2_main() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = ROOT / "results" / "continual_mgclap_phase2"
    for backbone in ["laion", "msclap"]:
        for seed in [0, 1, 2, 3]:
            summary_path = base / backbone / f"seed_{seed}" / "summary.json"
            data = load_json(summary_path)
            if not isinstance(data, dict):
                continue
            args = data.get("args", {})
            class_order = args.get("class_order")
            for row in data.get("aggregate_mean_std", []):
                method = row["method"]
                beta = float(row["beta"])
                if method in {"mgc_only", "mgp_mgc"} and beta != 4.0:
                    continue
                if method not in {"mgc_only", "mgp_mgc"} and beta != 0.0:
                    continue
                rows.append(
                    {
                        "source": str(summary_path),
                        "backbone": backbone,
                        "seed": seed,
                        "class_order": class_order,
                        "method": method,
                        "beta": beta,
                        "alpha": float(args.get("alpha", 0.10)),
                        "adapter_rank": int(args.get("adapter_rank", 16)),
                        "Avg_mean": float(row["Avg_mean"]),
                        "Avg_std": float(row["Avg_std"]),
                        "Last_mean": float(row["Last_mean"]),
                        "Last_std": float(row["Last_std"]),
                        "forgetting_proxy_mean": float(row["forgetting_proxy_mean"]),
                        "forgetting_proxy_std": float(row["forgetting_proxy_std"]),
                        "e_star_per_fold": data.get("e_star_per_fold", {}),
                        "summary_path": str(summary_path),
                    }
                )
    return rows


def collect_alpha_rank() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = ROOT / "results" / "ablations" / "laion_alpha_rank"
    if not base.exists():
        return rows
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        data = load_json(summary_path)
        if not isinstance(data, dict):
            continue
        args = data.get("args", {})
        for row in data.get("aggregate_mean_std", []):
            method = row["method"]
            beta = float(row["beta"])
            if method in {"mgc_only", "mgp_mgc"} and beta != 4.0:
                continue
            if method not in {"mgc_only", "mgp_mgc"} and beta != 0.0:
                continue
            rows.append(
                {
                    "source": str(summary_path),
                    "run_dir": run_dir.name,
                    "backbone": "laion",
                    "seed": int(args.get("seed", 0)),
                    "class_order": args.get("class_order"),
                    "method": method,
                    "beta": beta,
                    "alpha": float(args.get("alpha")),
                    "adapter_rank": int(args.get("adapter_rank")),
                    "Avg_mean": float(row["Avg_mean"]),
                    "Avg_std": float(row["Avg_std"]),
                    "Last_mean": float(row["Last_mean"]),
                    "Last_std": float(row["Last_std"]),
                    "forgetting_proxy_mean": float(row["forgetting_proxy_mean"]),
                    "forgetting_proxy_std": float(row["forgetting_proxy_std"]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def aggregate_over_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["backbone"], row["method"]), []).append(row)
    out = []
    for (backbone, method), vals in sorted(grouped.items()):
        out.append(
            {
                "backbone": backbone,
                "method": method,
                "runs": len(vals),
                "Avg_mean": float(np.mean([v["Avg_mean"] for v in vals])),
                "Avg_std": float(np.std([v["Avg_mean"] for v in vals])),
                "Last_mean": float(np.mean([v["Last_mean"] for v in vals])),
                "Last_std": float(np.std([v["Last_mean"] for v in vals])),
                "forgetting_proxy_mean": float(np.mean([v["forgetting_proxy_mean"] for v in vals])),
                "forgetting_proxy_std": float(np.std([v["forgetting_proxy_mean"] for v in vals])),
            }
        )
    return out


def class_order_robustness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for backbone in sorted({r["backbone"] for r in rows}):
        for method in sorted({r["method"] for r in rows}):
            subset = [r for r in rows if r["backbone"] == backbone and r["method"] == method]
            canonical = [r for r in subset if r["class_order"] == "canonical"]
            shuffled = [r for r in subset if r["class_order"] == "shuffled"]
            if not canonical or not shuffled:
                continue
            c = canonical[0]
            out.append(
                {
                    "backbone": backbone,
                    "method": method,
                    "canonical_seed": c["seed"],
                    "canonical_Avg": c["Avg_mean"],
                    "canonical_Last": c["Last_mean"],
                    "shuffled_runs": len(shuffled),
                    "shuffled_Avg_mean": float(np.mean([r["Avg_mean"] for r in shuffled])),
                    "shuffled_Avg_std": float(np.std([r["Avg_mean"] for r in shuffled])),
                    "shuffled_Last_mean": float(np.mean([r["Last_mean"] for r in shuffled])),
                    "shuffled_Last_std": float(np.std([r["Last_mean"] for r in shuffled])),
                    "delta_Avg_shuffled_minus_canonical": float(np.mean([r["Avg_mean"] for r in shuffled]) - c["Avg_mean"]),
                    "delta_Last_shuffled_minus_canonical": float(np.mean([r["Last_mean"] for r in shuffled]) - c["Last_mean"]),
                }
            )
    return out


def alpha_sensitivity_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], float(row["alpha"])), []).append(row)
    out = []
    for (method, alpha), vals in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "alpha": alpha,
                "runs": len(vals),
                "Avg_mean": float(np.mean([v["Avg_mean"] for v in vals])),
                "Avg_std": float(np.std([v["Avg_mean"] for v in vals])),
                "Last_mean": float(np.mean([v["Last_mean"] for v in vals])),
                "Last_std": float(np.std([v["Last_mean"] for v in vals])),
                "forgetting_proxy_mean": float(np.mean([v["forgetting_proxy_mean"] for v in vals])),
                "forgetting_proxy_std": float(np.std([v["forgetting_proxy_mean"] for v in vals])),
            }
        )
    return out


def rank_sensitivity_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], int(row["adapter_rank"])), []).append(row)
    out = []
    for (method, rank), vals in sorted(grouped.items()):
        out.append(
            {
                "method": method,
                "adapter_rank": rank,
                "runs": len(vals),
                "Avg_mean": float(np.mean([v["Avg_mean"] for v in vals])),
                "Avg_std": float(np.std([v["Avg_mean"] for v in vals])),
                "Last_mean": float(np.mean([v["Last_mean"] for v in vals])),
                "Last_std": float(np.std([v["Last_mean"] for v in vals])),
                "forgetting_proxy_mean": float(np.mean([v["forgetting_proxy_mean"] for v in vals])),
                "forgetting_proxy_std": float(np.std([v["forgetting_proxy_mean"] for v in vals])),
            }
        )
    return out


def plot_phase2_accuracy_by_seed(rows: list[dict[str, Any]], out_path: Path) -> None:
    methods = ["zero_shot_clap", "naive_adapter", "mgp_only", "mgc_only", "mgp_mgc"]
    colors = {
        "zero_shot_clap": "#6b7280",
        "naive_adapter": "#ef4444",
        "mgp_only": "#2563eb",
        "mgc_only": "#10b981",
        "mgp_mgc": "#7c3aed",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, backbone in zip(axes, ["laion", "msclap"]):
        subset = [r for r in rows if r["backbone"] == backbone]
        for method in methods:
            mrows = sorted([r for r in subset if r["method"] == method], key=lambda r: r["seed"])
            if not mrows:
                continue
            seeds = [r["seed"] for r in mrows]
            avg = [r["Avg_mean"] for r in mrows]
            ax.plot(seeds, avg, marker="o", lw=2, color=colors[method], label=method)
        ax.set_title(backbone)
        ax.set_xlabel("seed")
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Avg over tasks")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Phase 2 fixed-β=4 accuracy by seed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alpha_sensitivity(rows: list[dict[str, Any]], out_path: Path) -> None:
    target = [r for r in rows if r["method"] == "mgp_mgc"]
    zero_rows = [r for r in rows if r["method"] == "zero_shot_clap"]
    baseline = float(np.mean([r["Avg_mean"] for r in zero_rows])) if zero_rows else math.nan
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rank in sorted({int(r["adapter_rank"]) for r in target}):
        subset = sorted([r for r in target if int(r["adapter_rank"]) == rank], key=lambda r: r["alpha"])
        if not subset:
            continue
        ax.plot(
            [r["alpha"] for r in subset],
            [r["Avg_mean"] for r in subset],
            marker="o",
            lw=2,
            label=f"rank={rank}",
        )
    if not math.isnan(baseline):
        ax.axhline(baseline, color="#6b7280", ls="--", lw=1.5, label="zero-shot baseline")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Avg over tasks")
    ax.set_title("LAION alpha sensitivity (mgp_mgc, β=4)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_sensitivity(rows: list[dict[str, Any]], out_path: Path) -> None:
    target = [r for r in rows if r["method"] == "mgp_mgc"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha in sorted({float(r["alpha"]) for r in target}):
        subset = sorted([r for r in target if float(r["alpha"]) == alpha], key=lambda r: r["adapter_rank"])
        if not subset:
            continue
        ax.plot(
            [r["adapter_rank"] for r in subset],
            [r["Avg_mean"] for r in subset],
            marker="o",
            lw=2,
            label=f"alpha={alpha:g}",
        )
    ax.set_xlabel("adapter rank")
    ax.set_ylabel("Avg over tasks")
    ax.set_title("LAION rank sensitivity (mgp_mgc, β=4)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_main_table(rows: list[dict[str, Any]], out_csv: Path, out_png: Path) -> None:
    write_csv(
        out_csv,
        rows,
        [
            "backbone",
            "method",
            "runs",
            "Avg_mean",
            "Avg_std",
            "Last_mean",
            "Last_std",
            "forgetting_proxy_mean",
            "forgetting_proxy_std",
        ],
    )
    fig, ax = plt.subplots(figsize=(10, 3 + 0.35 * len(rows)))
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "No Phase 2 runs found yet.", ha="center", va="center", fontsize=12)
    else:
        table_rows = [
            [
                row["backbone"],
                row["method"],
                row["runs"],
                fmt(row["Avg_mean"], row["Avg_std"]),
                fmt(row["Last_mean"], row["Last_std"]),
                fmt(row["forgetting_proxy_mean"], row["forgetting_proxy_std"]),
            ]
            for row in rows
        ]
        table = ax.table(
            cellText=table_rows,
            colLabels=["backbone", "method", "runs", "Avg", "Last", "forgetting"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
    ax.set_title("Phase 2 fixed-β=4 main table", pad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stronger_claims(main_rows: list[dict[str, Any]], alpha_rows: list[dict[str, Any]]) -> list[str]:
    claims = []
    if any(r["backbone"] == "laion" for r in main_rows) and any(r["backbone"] == "msclap" for r in main_rows):
        claims.append("Backbone-level conclusions are no longer tied to a single seed/order run.")
    if any(r["method"] == "mgp_mgc" for r in main_rows):
        claims.append("The fixed-β=4 combined preservation+compensation result can be assessed for robustness across seeds and shuffled class orders.")
    if alpha_rows:
        claims.append("LAION sensitivity to alpha and adapter rank can be discussed with real ablation data rather than a single default configuration.")
    return claims


def unsafe_claims(alpha_rows: list[dict[str, Any]]) -> list[str]:
    claims = [
        "Do not claim full CLAP continual fine-tuning; Phase 2 still evaluates a frozen-embedding adapter setup.",
        "Do not claim beta is validation-tuned; Phase 2 fixes beta=4 for robustness checks but does not introduce a held-out tuning split.",
        "Do not claim Figure 2 is real-data validated.",
    ]
    if not alpha_rows:
        claims.append("Do not claim alpha/rank robustness until the LAION alpha-rank ablation has actually been run.")
    return claims


def build_markdown(
    summary: dict[str, Any],
    main_seed_rows: list[dict[str, Any]],
    class_order_rows: list[dict[str, Any]],
    alpha_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    main_table_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# MG-CLAP Phase 2 Status",
        "",
        f"- Timestamp: `{summary['timestamp']}`",
        f"- Git commit: `{summary['git_commit']}`",
        f"- Main-grid runs found: `{summary['main_grid_run_count']}`",
        f"- Alpha/rank runs found: `{summary['alpha_rank_run_count']}`",
        "",
        "## What Was Run",
        "",
    ]
    lines.extend(summary["what_was_run"] or ["- No Phase 2 runs found."])
    lines.extend(["", "## What Remains To Run", ""])
    lines.extend(summary["what_remains_to_run"] or ["- Nothing missing from the requested Phase 2 set."])

    lines.extend(
        [
            "",
            "## Fixed-Beta=4 Main Table",
            "",
            "| backbone | method | runs | Avg | Last | forgetting |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in main_table_rows:
        lines.append(
            f"| {row['backbone']} | {row['method']} | {row['runs']} | "
            f"{fmt(row['Avg_mean'], row['Avg_std'])} | {fmt(row['Last_mean'], row['Last_std'])} | "
            f"{fmt(row['forgetting_proxy_mean'], row['forgetting_proxy_std'])} |"
        )

    lines.extend(
        [
            "",
            "## By Seed",
            "",
            "| backbone | seed | order | method | Avg | Last | forgetting |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(main_seed_rows, key=lambda r: (r["backbone"], r["seed"], r["method"])):
        lines.append(
            f"| {row['backbone']} | {row['seed']} | {row['class_order']} | {row['method']} | "
            f"{row['Avg_mean']:.4f} | {row['Last_mean']:.4f} | {row['forgetting_proxy_mean']:.4f} |"
        )

    lines.extend(["", "## Class-Order Robustness", "", "| backbone | method | canonical Avg | shuffled Avg | delta Avg | canonical Last | shuffled Last | delta Last |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in class_order_rows:
        lines.append(
            f"| {row['backbone']} | {row['method']} | {row['canonical_Avg']:.4f} | "
            f"{fmt(row['shuffled_Avg_mean'], row['shuffled_Avg_std'])} | {row['delta_Avg_shuffled_minus_canonical']:.4f} | "
            f"{row['canonical_Last']:.4f} | {fmt(row['shuffled_Last_mean'], row['shuffled_Last_std'])} | "
            f"{row['delta_Last_shuffled_minus_canonical']:.4f} |"
        )

    lines.extend(["", "## Alpha Sensitivity", "", "| method | alpha | runs | Avg | Last | forgetting |", "|---|---:|---:|---:|---:|---:|"])
    for row in alpha_rows:
        lines.append(
            f"| {row['method']} | {row['alpha']:.2f} | {row['runs']} | {fmt(row['Avg_mean'], row['Avg_std'])} | "
            f"{fmt(row['Last_mean'], row['Last_std'])} | {fmt(row['forgetting_proxy_mean'], row['forgetting_proxy_std'])} |"
        )

    lines.extend(["", "## Rank Sensitivity", "", "| method | rank | runs | Avg | Last | forgetting |", "|---|---:|---:|---:|---:|---:|"])
    for row in rank_rows:
        lines.append(
            f"| {row['method']} | {row['adapter_rank']} | {row['runs']} | {fmt(row['Avg_mean'], row['Avg_std'])} | "
            f"{fmt(row['Last_mean'], row['Last_std'])} | {fmt(row['forgetting_proxy_mean'], row['forgetting_proxy_std'])} |"
        )

    lines.extend(["", "## Which Claims Become Stronger", ""])
    lines.extend([f"- {x}" for x in summary["which_claims_become_stronger"]] or ["- None yet."])
    lines.extend(["", "## Which Claims Remain Unsafe", ""])
    lines.extend([f"- {x}" for x in summary["which_claims_remain_unsafe"]] or ["- None listed."])
    return "\n".join(lines) + "\n"


def main() -> int:
    main_rows = collect_phase2_main()
    alpha_rank_rows = collect_alpha_rank()

    main_table_rows = aggregate_over_seeds(main_rows)
    class_order_rows = class_order_robustness(main_rows)
    alpha_rows = alpha_sensitivity_table(alpha_rank_rows)
    rank_rows = rank_sensitivity_table(alpha_rank_rows)

    results_dir = ROOT / "results"
    write_csv(
        results_dir / "phase2_main_grid_table.csv",
        main_rows,
        [
            "backbone",
            "seed",
            "class_order",
            "method",
            "beta",
            "alpha",
            "adapter_rank",
            "Avg_mean",
            "Avg_std",
            "Last_mean",
            "Last_std",
            "forgetting_proxy_mean",
            "forgetting_proxy_std",
            "summary_path",
        ],
    )
    write_csv(
        results_dir / "phase2_class_order_robustness.csv",
        class_order_rows,
        list(class_order_rows[0].keys()) if class_order_rows else [
            "backbone",
            "method",
            "canonical_seed",
            "canonical_Avg",
            "canonical_Last",
            "shuffled_runs",
            "shuffled_Avg_mean",
            "shuffled_Avg_std",
            "shuffled_Last_mean",
            "shuffled_Last_std",
            "delta_Avg_shuffled_minus_canonical",
            "delta_Last_shuffled_minus_canonical",
        ],
    )
    write_csv(
        results_dir / "phase2_alpha_sensitivity.csv",
        alpha_rows,
        list(alpha_rows[0].keys()) if alpha_rows else ["method", "alpha", "runs", "Avg_mean", "Avg_std", "Last_mean", "Last_std", "forgetting_proxy_mean", "forgetting_proxy_std"],
    )
    write_csv(
        results_dir / "phase2_rank_sensitivity.csv",
        rank_rows,
        list(rank_rows[0].keys()) if rank_rows else ["method", "adapter_rank", "runs", "Avg_mean", "Avg_std", "Last_mean", "Last_std", "forgetting_proxy_mean", "forgetting_proxy_std"],
    )

    plot_phase2_accuracy_by_seed(main_rows, results_dir / "phase2_accuracy_by_seed.png")
    plot_alpha_sensitivity(alpha_rank_rows, results_dir / "alpha_sensitivity.png")
    plot_rank_sensitivity(alpha_rank_rows, results_dir / "rank_sensitivity.png")
    plot_main_table(
        main_table_rows,
        results_dir / "fixed_beta4_main_table.csv",
        results_dir / "fixed_beta4_main_table.png",
    )

    expected_main = 8
    expected_alpha_rank = 16
    missing_main = max(0, expected_main - len({(r["backbone"], r["seed"]) for r in main_rows}))
    missing_alpha_rank = max(0, expected_alpha_rank - len({(r["alpha"], r["adapter_rank"]) for r in alpha_rank_rows}))

    what_was_run = []
    if main_rows:
        what_was_run.append(f"- Phase 2 main grid: {len({(r['backbone'], r['seed']) for r in main_rows})}/8 runs found.")
    if alpha_rank_rows:
        what_was_run.append(f"- LAION alpha/rank ablation: {len({(r['alpha'], r['adapter_rank']) for r in alpha_rank_rows})}/16 runs found.")
    what_remains = []
    if missing_main:
        what_remains.append(f"- Main-grid runs missing: {missing_main}")
    if missing_alpha_rank:
        what_remains.append(f"- Alpha/rank runs missing: {missing_alpha_rank}")

    summary = {
        "timestamp": utc_timestamp(),
        "git_commit": get_git_commit(ROOT),
        "command": command_string(),
        "machine_info": machine_info(),
        "main_grid_run_count": len({(r["backbone"], r["seed"]) for r in main_rows}),
        "alpha_rank_run_count": len({(r["alpha"], r["adapter_rank"]) for r in alpha_rank_rows}),
        "main_grid_rows": main_rows,
        "main_grid_aggregate": main_table_rows,
        "class_order_robustness": class_order_rows,
        "alpha_sensitivity": alpha_rows,
        "rank_sensitivity": rank_rows,
        "plots": {
            "phase2_accuracy_by_seed": file_info(results_dir / "phase2_accuracy_by_seed.png"),
            "alpha_sensitivity": file_info(results_dir / "alpha_sensitivity.png"),
            "rank_sensitivity": file_info(results_dir / "rank_sensitivity.png"),
            "fixed_beta4_main_table_png": file_info(results_dir / "fixed_beta4_main_table.png"),
            "fixed_beta4_main_table_csv": file_info(results_dir / "fixed_beta4_main_table.csv"),
        },
        "what_was_run": what_was_run,
        "what_remains_to_run": what_remains,
        "which_claims_become_stronger": stronger_claims(main_rows, alpha_rank_rows),
        "which_claims_remain_unsafe": unsafe_claims(alpha_rank_rows),
    }

    (results_dir / "PHASE2_STATUS.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "PHASE2_STATUS.md").write_text(
        build_markdown(summary, main_rows, class_order_rows, alpha_rows, rank_rows, main_table_rows)
    )

    print("What was run:")
    for line in summary["what_was_run"] or ["- No Phase 2 runs found."]:
        print(line)
    print("What remains to run:")
    for line in summary["what_remains_to_run"] or ["- Nothing missing from the requested Phase 2 set."]:
        print(line)
    print("Which claims become stronger:")
    for line in summary["which_claims_become_stronger"] or ["- None yet."]:
        print(f"- {line}" if not str(line).startswith("-") else line)
    print("Which claims remain unsafe:")
    for line in summary["which_claims_remain_unsafe"]:
        print(f"- {line}" if not str(line).startswith("-") else line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
