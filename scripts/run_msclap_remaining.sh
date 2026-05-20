#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/results/msclap_remaining.log"

mkdir -p "$ROOT/results"
: > "$LOG"

TASK_NAMES=(
  "MSCLAP Continual"
  "MSCLAP Shift"
  "MSCLAP Figure3 Real"
  "Project Summary"
)

TASK_CMDS=(
  "$PY table_1_zero_shot/continual_mgclap.py --backbone msclap --fold all --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 0 1 2 4 8 --seed 0"
  "$PY table_1_zero_shot/run.py --mode shift --backbone msclap"
  "$PY figure_3_contrastive_learning/run.py --backbone msclap --real"
  "$PY scripts/summarize_mgclap_project.py"
)

TOTAL="${#TASK_NAMES[@]}"
DONE=0

print_progress() {
  local done="$1"
  local total="$2"
  local width=32
  local filled=$(( done * width / total ))
  local empty=$(( width - filled ))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
  printf '\n[%s] %d/%d\n' "$bar" "$done" "$total"
  local i
  for ((i=0; i<total; i++)); do
    if (( i < done )); then
      printf '%02d. done     %s\n' "$((i + 1))" "${TASK_NAMES[$i]}"
    elif (( i == done )); then
      printf '%02d. running  %s\n' "$((i + 1))" "${TASK_NAMES[$i]}"
    else
      printf '%02d. pending  %s\n' "$((i + 1))" "${TASK_NAMES[$i]}"
    fi
  done
  printf '\n'
}

run_task() {
  local idx="$1"
  local name="${TASK_NAMES[$idx]}"
  local cmd="${TASK_CMDS[$idx]}"
  local start_ts
  start_ts="$(date -Iseconds)"

  {
    echo "================================================================================"
    echo "[$start_ts] START $name"
    echo "COMMAND: $cmd"
    echo "================================================================================"
  } | tee -a "$LOG"

  (
    cd "$ROOT"
    eval "$cmd"
  ) 2>&1 | tee -a "$LOG"

  local end_ts
  end_ts="$(date -Iseconds)"
  {
    echo
    echo "[$end_ts] END $name (exit=0)"
    echo
  } | tee -a "$LOG"
}

echo "Repository root: $ROOT"
echo "Using Python: $PY"
echo "Combined log: $LOG"

for ((i=0; i<TOTAL; i++)); do
  print_progress "$DONE" "$TOTAL"
  run_task "$i"
  DONE=$((DONE + 1))
done

print_progress "$DONE" "$TOTAL"
echo "MSCLAP run complete."
