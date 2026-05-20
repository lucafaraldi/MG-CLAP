# Codex Context

## Project State

This repo has moved from mostly scaffolding/synthetic checks to a real
presentation-safe MG-CLAP result set.

Real results now exist for both `laion` and `msclap` on:

- Figure 1 real AudioCaps modality-gap stats
- real ESC-50 shift sweeps
- real Figure 3 gap/temperature experiments
- real MG-CLAP-lite continual-learning runs

Phase 2 robustness is also complete:

- fixed-`beta=4` seed/class-order grid
- LAION alpha sensitivity
- LAION adapter-rank sensitivity

## Main Scientific Takeaways

- CLAP embeddings show a real modality gap on AudioCaps.
- LAION and MSCLAP differ substantially in geometry:
  - LAION gap is smaller
  - MSCLAP gap is larger and zero-shot is stronger
- Real shift helps clearly on LAION and only marginally on MSCLAP.
- In continual learning, naive sequential adapter training is harmful.
- Preservation plus audio-space compensation (`mgp_mgc`) is the strongest
  continual method on both backbones.
- Preservation alone is mixed and backbone-dependent.

## Key Result Files

Project status:

- `results/PROJECT_STATUS.md`
- `results/PHASE2_STATUS.md`

Real Figure 1:

- `results/figure_1/laion-audiocaps-val/stats.json`
- `results/figure_1/msclap-audiocaps-val/stats.json`

Real shift:

- `results/table_1/laion-shift/summary.json`
- `results/table_1/msclap-shift/summary.json`

Real Figure 3:

- `results/figure_3/laion-audiocaps-val/3a_gap_stats.json`
- `results/figure_3/laion-audiocaps-val/3b_landscape.json`
- `results/figure_3/msclap-audiocaps-val/3a_gap_stats.json`
- `results/figure_3/msclap-audiocaps-val/3b_landscape.json`

Continual learning:

- `results/continual_mgclap/laion/summary.json`
- `results/continual_mgclap/msclap/summary.json`
- `results/continual_mgclap/laion/RUN_REPORT.md`
- `results/continual_mgclap/msclap/RUN_REPORT.md`

Phase 2 robustness:

- `results/fixed_beta4_main_table.csv`
- `results/phase2_main_grid_table.csv`
- `results/phase2_class_order_robustness.csv`
- `results/phase2_alpha_sensitivity.csv`
- `results/phase2_rank_sensitivity.csv`
- `results/phase2_accuracy_by_seed.png`
- `results/alpha_sensitivity.png`
- `results/rank_sensitivity.png`

## Non-Presentation-Safe Artifacts

Do not use these as primary evidence:

- `results/figure_2/*`
- `results/figure_3/synth-gap0.82/*`
- `results/table_1/simulation/*`
- `results/table_1_training/laion/*`
- `results/continual_mgclap/laion/dry_run/*`

See also:

- `results/ARCHIVE_NOTE.md`

## Core Implementation Files

- `table_1_zero_shot/continual_mgclap.py`
- `table_1_zero_shot/run.py`
- `figure_3_contrastive_learning/run.py`
- `scripts/run_mgclap_pipeline.py`
- `scripts/run_msclap_remaining.sh`
- `scripts/summarize_mgclap_project.py`
- `scripts/run_mgclap_ablation_grid.py`
- `scripts/summarize_mgclap_ablation_grid.py`
- `lib/clap_models.py`
- `lib/gap_utils.py`

## Reproduction Commands

Main real pipeline:

```bash
./venv/bin/python scripts/run_mgclap_pipeline.py --include-msclap
```

Phase 2 robustness:

```bash
./venv/bin/python scripts/run_mgclap_ablation_grid.py --phase all
./venv/bin/python scripts/summarize_mgclap_ablation_grid.py
```

## Current Caveats

- This is a frozen-embedding adapter study, not full CLAP continual fine-tuning.
- `beta` is still exploratory rather than validation-tuned.
- Figure 2 is not real-data validated yet.
- New real shift/Figure 3 runs use the currently available AudioCaps subset
  rather than the full historical 1290-pair artifact used in the older Figure 1.
