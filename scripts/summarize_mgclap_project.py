#!/usr/bin/env python3
"""Summarize current MG-CLAP project state and classify saved result artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.provenance import command_string, file_info, get_git_commit, machine_info, utc_timestamp  # noqa: E402


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def classify_figure_1(results_root: Path) -> list[dict[str, Any]]:
    out = []
    for bb in ["laion-audiocaps-val", "msclap-audiocaps-val"]:
        stats = results_root / "figure_1" / bb / "stats.json"
        out.append(
            {
                "name": f"figure_1/{bb}",
                "status": "real-data" if stats.exists() else "missing",
                "safe_for_presentation": bool(stats.exists()),
                "notes": "Real AudioCaps gap stats." if stats.exists() else "Missing stats.json.",
                "files": [file_info(stats)],
            }
        )
    return out


def classify_figure_3(results_root: Path) -> list[dict[str, Any]]:
    out = []
    figure_3 = results_root / "figure_3"
    if not figure_3.exists():
        return [{"name": "figure_3", "status": "missing", "safe_for_presentation": False, "notes": "No figure_3 directory.", "files": []}]
    for child in sorted(p for p in figure_3.iterdir() if p.is_dir()):
        status = "synthetic" if child.name.startswith("synth-") else "real-data"
        safe = status == "real-data"
        out.append(
            {
                "name": f"figure_3/{child.name}",
                "status": status,
                "safe_for_presentation": safe,
                "notes": "Synthetic Figure 3 run." if status == "synthetic" else "Real-cache Figure 3 run.",
                "files": [file_info(child / "3a_gap_stats.json"), file_info(child / "3b_landscape.json")],
            }
        )
    return out


def classify_table_1(results_root: Path) -> list[dict[str, Any]]:
    out = []
    sim = results_root / "table_1" / "simulation" / "summary.json"
    out.append(
        {
            "name": "table_1/simulation",
            "status": "synthetic" if sim.exists() else "missing",
            "safe_for_presentation": False,
            "notes": "Synthetic zero-shot shift sweep." if sim.exists() else "Missing simulation summary.",
            "files": [file_info(sim)],
        }
    )
    for bb in ["laion-shift", "msclap-shift"]:
        path = results_root / "table_1" / bb / "summary.json"
        data = load_json(path)
        status = "missing"
        safe = False
        notes = "Missing real shift summary."
        if isinstance(data, dict):
            mode = data.get("mode")
            if mode == "real":
                status = "real-data"
                safe = True
                notes = "Real ESC-50 shift run."
            elif mode == "synthetic":
                status = "synthetic"
                notes = "Synthetic fallback summary."
            else:
                status = "unclear-provenance"
                notes = "Summary exists but lacks explicit provenance."
        out.append(
            {
                "name": f"table_1/{bb}",
                "status": status,
                "safe_for_presentation": safe,
                "notes": notes,
                "files": [file_info(path)],
            }
        )
    return out


def classify_table_1_training(results_root: Path) -> list[dict[str, Any]]:
    out = []
    base = results_root / "table_1_training"
    if not base.exists():
        return [{"name": "table_1_training", "status": "missing", "safe_for_presentation": False, "notes": "No table_1_training directory.", "files": []}]
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        summary = child / "sweep.json"
        data = load_json(summary)
        status = "unclear-provenance"
        notes = "Training sweep exists but provenance may be mixed or absent."
        safe = False
        if isinstance(data, dict) and data.get("mode") in {"real", "synthetic"}:
            status = "real-data" if data["mode"] == "real" else "synthetic"
            notes = f"Training sweep tagged as {data['mode']}."
            safe = data["mode"] == "real"
        out.append(
            {
                "name": f"table_1_training/{child.name}",
                "status": status,
                "safe_for_presentation": safe,
                "notes": notes,
                "files": [file_info(summary)],
            }
        )
    return out


def classify_continual(results_root: Path) -> list[dict[str, Any]]:
    out = []
    base = results_root / "continual_mgclap"
    if not base.exists():
        return [{"name": "continual_mgclap", "status": "missing", "safe_for_presentation": False, "notes": "No continual results yet.", "files": []}]
    for bb in sorted(p for p in base.iterdir() if p.is_dir()):
        for maybe_run in [bb] + [p for p in bb.iterdir() if p.is_dir()]:
            summary_path = maybe_run / "summary.json"
            if not summary_path.exists():
                continue
            data = load_json(summary_path) or {}
            status = "real-data"
            safe = True
            notes = "MG-CLAP-lite continual-learning run."
            if maybe_run.name == "dry_run":
                notes = "Dry run only."
            out.append(
                {
                    "name": str(maybe_run.relative_to(results_root)).replace("\\", "/"),
                    "status": status,
                    "safe_for_presentation": safe and maybe_run.name != "dry_run",
                    "notes": notes,
                    "files": [file_info(summary_path)],
                }
            )
    return out


def flatten(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for g in groups:
        out.extend(g)
    return out


def build_summary(results_root: Path) -> dict[str, Any]:
    items = flatten(
        [
            classify_figure_1(results_root),
            classify_figure_3(results_root),
            classify_table_1(results_root),
            classify_table_1_training(results_root),
            classify_continual(results_root),
        ]
    )
    real_esc50_continual = any(
        item["name"].startswith("continual_mgclap/") and item["status"] == "real-data"
        and not item["name"].endswith("/dry_run")
        for item in items
    )
    real_esc50_shift = any(
        item["name"] in {"table_1/laion-shift", "table_1/msclap-shift"} and item["status"] == "real-data"
        for item in items
    )
    real_clap_fig3 = any(
        item["name"].startswith("figure_3/") and item["status"] == "real-data"
        for item in items
    )
    safe = [item["name"] for item in items if item["safe_for_presentation"]]
    unsafe = [item["name"] for item in items if not item["safe_for_presentation"]]
    missing = [item["name"] for item in items if item["status"] == "missing"]
    return {
        "timestamp": utc_timestamp(),
        "git_commit": get_git_commit(ROOT),
        "command": command_string(),
        "machine_info": machine_info(),
        "items": items,
        "safe_for_presentation": safe,
        "not_safe_for_presentation": unsafe,
        "missing_results": missing,
        "real_esc50_continual_learning_exists": real_esc50_continual,
        "real_esc50_shift_exists": real_esc50_shift,
        "real_clap_figure3_exists": real_clap_fig3,
        "what_remains_to_run": [
            item["name"] for item in items if item["status"] in {"missing", "synthetic", "unclear-provenance"}
        ],
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MG-CLAP Project Status",
        "",
        f"- Timestamp: `{summary['timestamp']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Result Classification",
        "",
        "| result | status | safe? | notes |",
        "|---|---|---|---|",
    ]
    for item in summary["items"]:
        lines.append(
            f"| {item['name']} | {item['status']} | "
            f"{'yes' if item['safe_for_presentation'] else 'no'} | {item['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Presentation Safety",
            "",
            "### Safe for Presentation",
            "",
        ]
    )
    lines.extend([f"- {name}" for name in summary["safe_for_presentation"]] or ["- None"])
    lines.extend(
        [
            "",
            "### Not Safe for Presentation",
            "",
        ]
    )
    lines.extend([f"- {name}" for name in summary["not_safe_for_presentation"]] or ["- None"])
    lines.extend(
        [
            "",
            "## Explicit Answers",
            "",
            f"- Real ESC-50 continual learning exists: `{summary['real_esc50_continual_learning_exists']}`",
            f"- Real ESC-50 shift exists: `{summary['real_esc50_shift_exists']}`",
            f"- Real CLAP Figure 3 exists: `{summary['real_clap_figure3_exists']}`",
            "",
            "## Remaining Work",
            "",
        ]
    )
    lines.extend([f"- {name}" for name in summary["what_remains_to_run"]] or ["- Nothing classified as pending."])
    return "\n".join(lines) + "\n"


def main() -> None:
    results_root = ROOT / "results"
    results_root.mkdir(exist_ok=True)
    summary = build_summary(results_root)
    (results_root / "PROJECT_STATUS.json").write_text(json.dumps(summary, indent=2))
    (results_root / "PROJECT_STATUS.md").write_text(build_markdown(summary))
    print(f"Wrote {results_root / 'PROJECT_STATUS.md'}")
    print(f"Wrote {results_root / 'PROJECT_STATUS.json'}")


if __name__ == "__main__":
    main()
