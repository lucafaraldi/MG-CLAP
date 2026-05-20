# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold all --seed 0 --tasks 10 --classes-per-task 5 --alpha 0.1 --adapter-rank 32 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order canonical --out-dir /tmp/mgclap_phase2/alpha_rank/alpha_0p10__rank_32`
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
| mgc_only | 4.0 | 1 | 0.7357 | 0.7100 | 0.3056 | 3 |
| mgc_only | 4.0 | 2 | 0.7180 | 0.6550 | 0.3750 | 3 |
| mgc_only | 4.0 | 3 | 0.7445 | 0.6675 | 0.3556 | 3 |
| mgc_only | 4.0 | 4 | 0.7320 | 0.6800 | 0.3389 | 3 |
| mgc_only | 4.0 | 5 | 0.7010 | 0.6350 | 0.3972 | 3 |
| mgp_mgc | 4.0 | 1 | 0.9465 | 0.9250 | 0.0667 | 3 |
| mgp_mgc | 4.0 | 2 | 0.9654 | 0.9250 | 0.0750 | 3 |
| mgp_mgc | 4.0 | 3 | 0.9685 | 0.9200 | 0.0806 | 3 |
| mgp_mgc | 4.0 | 4 | 0.9548 | 0.9075 | 0.0778 | 3 |
| mgp_mgc | 4.0 | 5 | 0.9298 | 0.8825 | 0.1111 | 3 |
| mgp_only | 0.0 | 1 | 0.8229 | 0.8325 | 0.1611 | 3 |
| mgp_only | 0.0 | 2 | 0.8161 | 0.7825 | 0.2139 | 3 |
| mgp_only | 0.0 | 3 | 0.8441 | 0.8300 | 0.1583 | 3 |
| mgp_only | 0.0 | 4 | 0.8417 | 0.8025 | 0.1861 | 3 |
| mgp_only | 0.0 | 5 | 0.8196 | 0.8150 | 0.1639 | 3 |
| naive_adapter | 0.0 | 1 | 0.6787 | 0.5925 | 0.4389 | 3 |
| naive_adapter | 0.0 | 2 | 0.6730 | 0.5375 | 0.5028 | 3 |
| naive_adapter | 0.0 | 3 | 0.7150 | 0.6225 | 0.4139 | 3 |
| naive_adapter | 0.0 | 4 | 0.6902 | 0.6425 | 0.3806 | 3 |
| naive_adapter | 0.0 | 5 | 0.6771 | 0.5450 | 0.5000 | 3 |
| zero_shot_clap | 0.0 | 1 | 0.8187 | 0.8425 | 0.0389 | 3 |
| zero_shot_clap | 0.0 | 2 | 0.8184 | 0.8200 | 0.0556 | 3 |
| zero_shot_clap | 0.0 | 3 | 0.8197 | 0.8275 | 0.0472 | 3 |
| zero_shot_clap | 0.0 | 4 | 0.8094 | 0.8375 | 0.0306 | 3 |
| zero_shot_clap | 0.0 | 5 | 0.8217 | 0.8275 | 0.0500 | 3 |

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
