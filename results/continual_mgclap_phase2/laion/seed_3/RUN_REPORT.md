# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold all --seed 3 --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order shuffled --out-dir /tmp/mgclap_phase2/main_grid/laion/seed_3`
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
| mgc_only | 4.0 | 1 | 0.8283 | 0.6675 | 0.3639 | 3 |
| mgc_only | 4.0 | 2 | 0.8339 | 0.6650 | 0.3667 | 3 |
| mgc_only | 4.0 | 3 | 0.8308 | 0.6850 | 0.3417 | 2 |
| mgc_only | 4.0 | 4 | 0.8452 | 0.6950 | 0.3333 | 2 |
| mgc_only | 4.0 | 5 | 0.7990 | 0.6400 | 0.3889 | 2 |
| mgp_mgc | 4.0 | 1 | 0.9601 | 0.9025 | 0.0917 | 3 |
| mgp_mgc | 4.0 | 2 | 0.9666 | 0.9175 | 0.0722 | 3 |
| mgp_mgc | 4.0 | 3 | 0.9755 | 0.9375 | 0.0611 | 2 |
| mgp_mgc | 4.0 | 4 | 0.9751 | 0.9500 | 0.0444 | 2 |
| mgp_mgc | 4.0 | 5 | 0.9595 | 0.9150 | 0.0833 | 2 |
| mgp_only | 0.0 | 1 | 0.9215 | 0.8775 | 0.0972 | 3 |
| mgp_only | 0.0 | 2 | 0.9349 | 0.8725 | 0.1028 | 3 |
| mgp_only | 0.0 | 3 | 0.9457 | 0.9000 | 0.0806 | 2 |
| mgp_only | 0.0 | 4 | 0.9414 | 0.9025 | 0.0667 | 2 |
| mgp_only | 0.0 | 5 | 0.9261 | 0.8900 | 0.0861 | 2 |
| naive_adapter | 0.0 | 1 | 0.7972 | 0.6925 | 0.3389 | 3 |
| naive_adapter | 0.0 | 2 | 0.7928 | 0.6325 | 0.4056 | 3 |
| naive_adapter | 0.0 | 3 | 0.7805 | 0.6350 | 0.3972 | 2 |
| naive_adapter | 0.0 | 4 | 0.8081 | 0.6475 | 0.3861 | 2 |
| naive_adapter | 0.0 | 5 | 0.7878 | 0.6350 | 0.4028 | 2 |
| zero_shot_clap | 0.0 | 1 | 0.8989 | 0.8425 | 0.0222 | 3 |
| zero_shot_clap | 0.0 | 2 | 0.8803 | 0.8200 | 0.0333 | 3 |
| zero_shot_clap | 0.0 | 3 | 0.8978 | 0.8275 | 0.0472 | 2 |
| zero_shot_clap | 0.0 | 4 | 0.9014 | 0.8375 | 0.0333 | 2 |
| zero_shot_clap | 0.0 | 5 | 0.8905 | 0.8275 | 0.0444 | 2 |

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
