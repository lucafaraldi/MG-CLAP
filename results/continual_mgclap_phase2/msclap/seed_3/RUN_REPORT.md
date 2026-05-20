# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone msclap --fold all --seed 3 --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order shuffled --out-dir /tmp/mgclap_phase2/main_grid/msclap/seed_3`
- Backbone: `msclap`
- Folds: `[1, 2, 3, 4, 5]`
- Tasks x classes/task: `10 x 5`
- Dry run: `False`

## What Was Implemented

- Zero-shot CLAP baseline
- Sequential low-rank residual audio adapter
- Modality-gap preservation epoch selection via task-1 negative-similarity drift
- Audio prototype compensation head with explicit beta sweep
- Fold-wise summary, trajectories, and provenance metadata

## What Remains Synthetic Elsewhere In The Repo

- Current saved Figure 2 outputs are still synthetic/random-init stand-ins.
- Current saved Figure 3 outputs may still be synthetic unless re-run with `--real`.
- Current saved Table 1 shift outputs may still be synthetic unless re-run in shift mode with real caches.

## Main Table

| method | beta | fold | Avg | Last | forgetting_proxy | e_star |
|---|---:|---:|---:|---:|---:|---:|
| mgc_only | 4.0 | 1 | 0.8517 | 0.8100 | 0.1972 | 1 |
| mgc_only | 4.0 | 2 | 0.8679 | 0.8025 | 0.2139 | 1 |
| mgc_only | 4.0 | 3 | 0.8684 | 0.8250 | 0.1861 | 1 |
| mgc_only | 4.0 | 4 | 0.8904 | 0.8475 | 0.1694 | 1 |
| mgc_only | 4.0 | 5 | 0.8504 | 0.8125 | 0.2028 | 1 |
| mgp_mgc | 4.0 | 1 | 0.9810 | 0.9775 | 0.0139 | 1 |
| mgp_mgc | 4.0 | 2 | 0.9892 | 0.9850 | 0.0083 | 1 |
| mgp_mgc | 4.0 | 3 | 0.9844 | 0.9625 | 0.0306 | 1 |
| mgp_mgc | 4.0 | 4 | 0.9922 | 0.9875 | 0.0083 | 1 |
| mgp_mgc | 4.0 | 5 | 0.9671 | 0.9575 | 0.0333 | 1 |
| mgp_only | 0.0 | 1 | 0.9612 | 0.9525 | 0.0389 | 1 |
| mgp_only | 0.0 | 2 | 0.9778 | 0.9675 | 0.0194 | 1 |
| mgp_only | 0.0 | 3 | 0.9849 | 0.9625 | 0.0306 | 1 |
| mgp_only | 0.0 | 4 | 0.9851 | 0.9575 | 0.0417 | 1 |
| mgp_only | 0.0 | 5 | 0.9607 | 0.9400 | 0.0500 | 1 |
| naive_adapter | 0.0 | 1 | 0.8136 | 0.8000 | 0.2111 | 1 |
| naive_adapter | 0.0 | 2 | 0.8495 | 0.8225 | 0.1917 | 1 |
| naive_adapter | 0.0 | 3 | 0.8230 | 0.7475 | 0.2722 | 1 |
| naive_adapter | 0.0 | 4 | 0.8489 | 0.7925 | 0.2278 | 1 |
| naive_adapter | 0.0 | 5 | 0.8126 | 0.7575 | 0.2639 | 1 |
| zero_shot_clap | 0.0 | 1 | 0.9671 | 0.9400 | 0.0306 | 1 |
| zero_shot_clap | 0.0 | 2 | 0.9752 | 0.9650 | 0.0083 | 1 |
| zero_shot_clap | 0.0 | 3 | 0.9696 | 0.9325 | 0.0333 | 1 |
| zero_shot_clap | 0.0 | 4 | 0.9801 | 0.9475 | 0.0333 | 1 |
| zero_shot_clap | 0.0 | 5 | 0.9625 | 0.9325 | 0.0333 | 1 |

## Interpretation

- `mgp_only` tests whether limiting drift in negative audio-text similarity stabilizes continual learning.
- `mgc_only` tests whether audio-space prototypes compensate for text-only classifier limits.
- `mgp_mgc` tests the combination. Beta is an exploratory sweep, not a hidden validation-tuned parameter.

## Limitations

- Frozen-embedding MG-CLAP-lite is not full CLAP fine-tuning.
- No replay buffer is used.
- Beta sweep is exploratory and uses the evaluation folds directly in the current implementation summary.
- This run does not retroactively certify provenance of older result files elsewhere in the repo.

## Presentation-Safe Claims

- MG-CLAP-lite runs real ESC-50 continual-learning experiments only when real CLAP caches are present.
- The adapter starts as an identity-like residual map and does not silently fall back to synthetic data.
- The preservation rule is explicitly defined by task-1 negative audio-text similarity drift and recorded per fold.
- Prototype compensation is reported as an exploratory beta sweep, not as silently tuned test-time optimization.

## Claims To Avoid

- Do not claim this is full CLAP continual fine-tuning; it is a frozen-embedding adapter study.
- Do not claim beta was tuned on a held-out validation split in the current exploratory sweep.
- Do not claim synthetic Figure 2 or Figure 3 artifacts elsewhere in the repo are real without re-running them.
- Do not claim modality-gap preservation transfers unless the real continual-learning run is actually executed and inspected.

## Next Experiments

- Run all five ESC-50 folds for both backbones.
- Compare best-beta-by-Avg against the text-only continual baselines.
- Re-run real Table 1 shift and real Figure 3 with provenance-enabled summaries.
