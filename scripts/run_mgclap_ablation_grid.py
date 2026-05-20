#!/usr/bin/env python3
"""Phase 2 orchestration for MG-CLAP-lite robustness experiments."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.provenance import command_string, file_info, get_git_commit, machine_info, utc_timestamp  # noqa: E402


@dataclass
class Task:
    name: str
    command: list[str]
    final_dir: Path
    temp_out_dir: Path
    phase: str
    meta: dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=["main-grid", "alpha-rank", "all"],
        default="all",
        help="Which Phase 2 task family to run.",
    )
    p.add_argument(
        "--python",
        default=None,
        help="Python interpreter to use. Defaults to repo-local venv when present.",
    )
    p.add_argument(
        "--log-file",
        default=str(ROOT / "results" / "phase2_ablation_grid.log"),
        help="Combined log file for all ablation runs.",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print the commands that would run, but do not execute them.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later tasks after a failure.",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose final summary.json already exists.",
    )
    return p.parse_args()


def pick_python(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    repo_venv = ROOT / "venv" / "bin" / "python"
    if repo_venv.exists():
        return repo_venv
    return Path(sys.executable)


def slug_alpha(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def build_main_grid_tasks(py: Path) -> list[Task]:
    tasks: list[Task] = []
    for backbone in ["laion", "msclap"]:
        for seed in [0, 1, 2, 3]:
            class_order = "canonical" if seed == 0 else "shuffled"
            temp_out = Path("/tmp") / "mgclap_phase2" / "main_grid" / backbone / f"seed_{seed}"
            final_dir = ROOT / "results" / "continual_mgclap_phase2" / backbone / f"seed_{seed}"
            cmd = [
                str(py),
                "table_1_zero_shot/continual_mgclap.py",
                "--backbone",
                backbone,
                "--fold",
                "all",
                "--seed",
                str(seed),
                "--tasks",
                "10",
                "--classes-per-task",
                "5",
                "--alpha",
                "0.10",
                "--adapter-rank",
                "16",
                "--epochs-max",
                "20",
                "--lr",
                "1e-3",
                "--weight-decay",
                "1e-4",
                "--batch-size",
                "64",
                "--betas",
                "4",
                "--class-order",
                class_order,
                "--out-dir",
                str(temp_out),
            ]
            tasks.append(
                Task(
                    name=f"MainGrid {backbone} seed={seed} order={class_order}",
                    command=cmd,
                    final_dir=final_dir,
                    temp_out_dir=temp_out / backbone,
                    phase="main-grid",
                    meta={
                        "backbone": backbone,
                        "seed": seed,
                        "class_order": class_order,
                        "beta_fixed": 4.0,
                        "alpha": 0.10,
                        "adapter_rank": 16,
                    },
                )
            )
    return tasks


def build_alpha_rank_tasks(py: Path) -> list[Task]:
    tasks: list[Task] = []
    for alpha in [0.05, 0.10, 0.20, 0.30]:
        for rank in [4, 8, 16, 32]:
            slug = f"alpha_{slug_alpha(alpha)}__rank_{rank:02d}"
            temp_out = Path("/tmp") / "mgclap_phase2" / "alpha_rank" / slug
            final_dir = ROOT / "results" / "ablations" / "laion_alpha_rank" / slug
            cmd = [
                str(py),
                "table_1_zero_shot/continual_mgclap.py",
                "--backbone",
                "laion",
                "--fold",
                "all",
                "--seed",
                "0",
                "--tasks",
                "10",
                "--classes-per-task",
                "5",
                "--alpha",
                str(alpha),
                "--adapter-rank",
                str(rank),
                "--epochs-max",
                "20",
                "--lr",
                "1e-3",
                "--weight-decay",
                "1e-4",
                "--batch-size",
                "64",
                "--betas",
                "4",
                "--class-order",
                "canonical",
                "--out-dir",
                str(temp_out),
            ]
            tasks.append(
                Task(
                    name=f"AlphaRank laion alpha={alpha:.2f} rank={rank}",
                    command=cmd,
                    final_dir=final_dir,
                    temp_out_dir=temp_out / "laion",
                    phase="alpha-rank",
                    meta={
                        "backbone": "laion",
                        "seed": 0,
                        "class_order": "canonical",
                        "beta_fixed": 4.0,
                        "alpha": alpha,
                        "adapter_rank": rank,
                    },
                )
            )
    return tasks


def build_tasks(args: argparse.Namespace, py: Path) -> list[Task]:
    tasks: list[Task] = []
    if args.phase in {"main-grid", "all"}:
        tasks.extend(build_main_grid_tasks(py))
    if args.phase in {"alpha-rank", "all"}:
        tasks.extend(build_alpha_rank_tasks(py))
    return tasks


def progress_bar(done: int, total: int, width: int = 32) -> str:
    filled = 0 if total == 0 else int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def copy_result_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Expected temp result directory missing: {src}")
    if (dst / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def run_task(task: Task, log_fh) -> int:
    start = time.time()
    header = (
        "\n"
        + "=" * 80
        + f"\n[{datetime.now().isoformat()}] START {task.name}\n"
        + f"COMMAND: {' '.join(task.command)}\n"
        + f"FINAL_DIR: {task.final_dir}\n"
        + "=" * 80
        + "\n"
    )
    print(header, end="")
    log_fh.write(header)
    log_fh.flush()

    env = os.environ.copy()
    env["PATH"] = str(Path(task.command[0]).parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen(
        task.command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        log_fh.write(line)
    proc.wait()

    if proc.returncode == 0:
        copy_result_tree(task.temp_out_dir, task.final_dir)

    elapsed = time.time() - start
    footer = (
        f"\n[{datetime.now().isoformat()}] END {task.name} "
        f"(exit={proc.returncode}, elapsed={elapsed:.1f}s)\n"
    )
    print(footer, end="")
    log_fh.write(footer)
    log_fh.flush()
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    py = pick_python(args.python)
    tasks = build_tasks(args, py)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": utc_timestamp(),
        "git_commit": get_git_commit(ROOT),
        "command": command_string(),
        "machine_info": machine_info(),
        "python": str(py),
        "phase": args.phase,
        "task_count": len(tasks),
        "tasks": [
            {
                "name": t.name,
                "command": t.command,
                "phase": t.phase,
                "final_dir": str(t.final_dir),
                "temp_out_dir": str(t.temp_out_dir),
                "meta": t.meta,
            }
            for t in tasks
        ],
    }
    (ROOT / "results" / "phase2_ablation_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Repository root: {ROOT}")
    print(f"Using Python: {py}")
    print(f"Combined log: {log_path}")
    print(f"Phase: {args.phase}")
    print(f"Planned tasks: {len(tasks)}")

    if args.print_only:
        for i, task in enumerate(tasks, start=1):
            print(f"{i:02d}. {task.name}")
            print("    " + " ".join(task.command))
            print(f"    -> {task.final_dir}")
        print("\nWhat was run: nothing, print-only mode.")
        print("What remains to run:")
        for task in tasks:
            print(f"- {task.name}")
        print("Which claims become stronger: none yet; no new experiments executed.")
        print("Which claims remain unsafe: all Phase 2 robustness claims remain untested.")
        return 0

    statuses = ["pending"] * len(tasks)
    attempted = 0
    completed = 0
    skipped = 0
    failed: list[str] = []

    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n\n### Phase 2 run started {datetime.now().isoformat()} ###\n")
        for idx, task in enumerate(tasks):
            print()
            print(progress_bar(completed + skipped, len(tasks)))
            if args.skip_existing and (task.final_dir / "summary.json").exists():
                statuses[idx] = "skipped"
                skipped += 1
                print(f"{task.name}: skipping existing result at {task.final_dir}")
                continue
            statuses[idx] = "running"
            attempted += 1
            code = run_task(task, log_fh)
            if code == 0:
                statuses[idx] = "done"
                completed += 1
            else:
                statuses[idx] = "failed"
                failed.append(task.name)
                if not args.continue_on_error:
                    break

    print("\nWhat was run:")
    print(f"- attempted tasks: {attempted}")
    print(f"- completed tasks: {completed}")
    print(f"- skipped existing: {skipped}")
    if failed:
        print("- failed tasks:")
        for name in failed:
            print(f"  - {name}")

    remaining = [
        task.name
        for task in tasks
        if not (task.final_dir / "summary.json").exists()
    ]
    print("What remains to run:")
    if remaining:
        for name in remaining:
            print(f"- {name}")
    else:
        print("- Nothing in the requested Phase 2 grid.")

    if args.phase in {"main-grid", "all"} and completed + skipped >= 8:
        print("Which claims become stronger:")
        print("- Cross-seed and class-order robustness claims for the fixed-beta=4 MG-CLAP-lite setup.")
        print("- Whether LAION vs MSCLAP conclusions persist beyond a single seed/order.")
    else:
        print("Which claims become stronger:")
        print("- Only after the requested main-grid runs complete.")

    if args.phase in {"alpha-rank", "all"} and completed + skipped >= 16:
        print("- Alpha sensitivity and adapter-rank sensitivity claims for LAION.")

    print("Which claims remain unsafe:")
    print("- Any claim of robustness that has not yet been run across seeds/orders/ablations.")
    print("- Any claim that Figure 2 is real-data validated.")
    print("- Any claim that beta is cleanly validation-tuned rather than exploratory.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
