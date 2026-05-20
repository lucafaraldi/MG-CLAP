# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold all --seed 1 --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order shuffled --out-dir /tmp/mgclap_phase2/main_grid/laion/seed_1`
- Backbone: `laion`
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
| mgc_only | 4.0 | 1 | 0.7986 | 0.8150 | 0.2000 | 1 |
| mgc_only | 4.0 | 2 | 0.7831 | 0.7925 | 0.2278 | 1 |
| mgc_only | 4.0 | 3 | 0.8033 | 0.7900 | 0.2306 | 1 |
| mgc_only | 4.0 | 4 | 0.8361 | 0.8175 | 0.1944 | 1 |
| mgc_only | 4.0 | 5 | 0.8179 | 0.7675 | 0.2556 | 1 |
| mgp_mgc | 4.0 | 1 | 0.9663 | 0.9675 | 0.0278 | 1 |
| mgp_mgc | 4.0 | 2 | 0.9844 | 0.9775 | 0.0250 | 1 |
| mgp_mgc | 4.0 | 3 | 0.9805 | 0.9675 | 0.0222 | 1 |
| mgp_mgc | 4.0 | 4 | 0.9771 | 0.9625 | 0.0194 | 1 |
| mgp_mgc | 4.0 | 5 | 0.9815 | 0.9650 | 0.0167 | 1 |
| mgp_only | 0.0 | 1 | 0.9062 | 0.9150 | 0.0389 | 1 |
| mgp_only | 0.0 | 2 | 0.8720 | 0.8850 | 0.0667 | 1 |
| mgp_only | 0.0 | 3 | 0.9428 | 0.9050 | 0.0639 | 1 |
| mgp_only | 0.0 | 4 | 0.9194 | 0.8950 | 0.0556 | 1 |
| mgp_only | 0.0 | 5 | 0.9217 | 0.9075 | 0.0444 | 1 |
| naive_adapter | 0.0 | 1 | 0.7629 | 0.7825 | 0.2361 | 1 |
| naive_adapter | 0.0 | 2 | 0.7505 | 0.7700 | 0.2500 | 1 |
| naive_adapter | 0.0 | 3 | 0.7656 | 0.7175 | 0.3139 | 1 |
| naive_adapter | 0.0 | 4 | 0.7764 | 0.7750 | 0.2444 | 1 |
| naive_adapter | 0.0 | 5 | 0.7812 | 0.7800 | 0.2444 | 1 |
| zero_shot_clap | 0.0 | 1 | 0.8675 | 0.8425 | 0.0611 | 1 |
| zero_shot_clap | 0.0 | 2 | 0.8422 | 0.8200 | 0.0861 | 1 |
| zero_shot_clap | 0.0 | 3 | 0.8907 | 0.8275 | 0.0889 | 1 |
| zero_shot_clap | 0.0 | 4 | 0.8889 | 0.8375 | 0.0750 | 1 |
| zero_shot_clap | 0.0 | 5 | 0.8789 | 0.8275 | 0.0667 | 1 |

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
