#!/usr/bin/env python3
"""Sequential runner for the MG-CLAP command pipeline.

Runs the main experiment commands one after another, shows a simple progress bar,
and writes combined stdout/stderr for every task to a single log file.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Task:
    name: str
    command: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log-file",
        default=str(ROOT / "results" / "mgclap_pipeline.log"),
        help="Single combined log file for stdout/stderr.",
    )
    p.add_argument(
        "--include-msclap",
        action="store_true",
        help="Include optional MSCLAP extraction and experiment commands.",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip embedding extraction commands.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed task instead of stopping immediately.",
    )
    p.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Run only the file check and the LAION continual dry-run.",
    )
    return p.parse_args()


def build_tasks(args: argparse.Namespace) -> list[Task]:
    py = shlex.quote(sys.executable)
    tasks = [
        Task("Check Files", "ls -R embeddings data results | head -300"),
    ]

    if not args.skip_extract:
        tasks.extend(
            [
                Task(
                    "Extract LAION AudioCaps",
                    f"{py} scripts/01_extract_embeddings.py --backbone laion --dataset audiocaps --split val",
                ),
                Task(
                    "Extract LAION ESC-50",
                    f"{py} scripts/01_extract_embeddings.py --backbone laion --dataset esc50",
                ),
            ]
        )
        if args.include_msclap:
            tasks.extend(
                [
                    Task(
                        "Extract MSCLAP AudioCaps",
                        f"{py} scripts/01_extract_embeddings.py --backbone msclap --dataset audiocaps --split val",
                    ),
                    Task(
                        "Extract MSCLAP ESC-50",
                        f"{py} scripts/01_extract_embeddings.py --backbone msclap --dataset esc50",
                    ),
                ]
            )

    tasks.append(
        Task(
            "Continual Dry Run LAION",
            f"{py} table_1_zero_shot/continual_mgclap.py --backbone laion --fold 1 --dry-run",
        )
    )

    if args.dry_run_only:
        return tasks

    tasks.extend(
        [
            Task(
                "Continual Full LAION",
                f"{py} table_1_zero_shot/continual_mgclap.py "
                "--backbone laion --fold all --tasks 10 --classes-per-task 5 "
                "--alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 "
                "--weight-decay 1e-4 --batch-size 64 --betas 0 1 2 4 8 --seed 0",
            ),
            Task(
                "Real Shift LAION",
                f"{py} table_1_zero_shot/run.py --mode shift --backbone laion",
            ),
            Task(
                "Real Figure 3 LAION",
                f"{py} figure_3_contrastive_learning/run.py --backbone laion --real",
            ),
        ]
    )

    if args.include_msclap:
        tasks.extend(
            [
                Task(
                    "Continual Full MSCLAP",
                    f"{py} table_1_zero_shot/continual_mgclap.py "
                    "--backbone msclap --fold all --tasks 10 --classes-per-task 5 "
                    "--alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 "
                    "--weight-decay 1e-4 --batch-size 64 --betas 0 1 2 4 8 --seed 0",
                ),
                Task(
                    "Real Shift MSCLAP",
                    f"{py} table_1_zero_shot/run.py --mode shift --backbone msclap",
                ),
                Task(
                    "Real Figure 3 MSCLAP",
                    f"{py} figure_3_contrastive_learning/run.py --backbone msclap --real",
                ),
            ]
        )

    tasks.append(Task("Summarize Project", f"{py} scripts/summarize_mgclap_project.py"))
    return tasks


def progress_bar(done: int, total: int, width: int = 32) -> str:
    filled = 0 if total == 0 else int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def print_status(tasks: list[Task], statuses: list[str]) -> None:
    done = sum(1 for s in statuses if s == "done")
    print()
    print(progress_bar(done, len(tasks)))
    for idx, (task, status) in enumerate(zip(tasks, statuses), start=1):
        print(f"{idx:02d}. {status:8s} {task.name}")
    print()


def run_task(task: Task, log_fh) -> int:
    start = time.time()
    header = (
        "\n"
        + "=" * 80
        + f"\n[{datetime.now().isoformat()}] START {task.name}\n"
        + f"COMMAND: {task.command}\n"
        + "=" * 80
        + "\n"
    )
    print(header, end="")
    log_fh.write(header)
    log_fh.flush()

    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

    proc = subprocess.Popen(
        ["/bin/bash", "-lc", task.command],
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
    tasks = build_tasks(args)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    statuses = ["pending"] * len(tasks)
    print(f"Repository root: {ROOT}")
    print(f"Using Python: {sys.executable}")
    print(f"Combined log: {log_path}")
    print_status(tasks, statuses)

    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(
            f"\n\n### Pipeline run started {datetime.now().isoformat()} ###\n"
        )
        for i, task in enumerate(tasks):
            statuses[i] = "running"
            print_status(tasks, statuses)
            code = run_task(task, log_fh)
            if code == 0:
                statuses[i] = "done"
            else:
                statuses[i] = "failed"
                print_status(tasks, statuses)
                if not args.continue_on_error:
                    print(f"Stopping on failure: {task.name}")
                    return code
            print_status(tasks, statuses)

        log_fh.write(
            f"### Pipeline run finished {datetime.now().isoformat()} ###\n"
        )

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
