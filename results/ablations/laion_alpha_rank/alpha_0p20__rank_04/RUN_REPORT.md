# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold all --seed 0 --tasks 10 --classes-per-task 5 --alpha 0.2 --adapter-rank 4 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order canonical --out-dir /tmp/mgclap_phase2/alpha_rank/alpha_0p20__rank_04`
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
| mgc_only | 4.0 | 1 | 0.8512 | 0.8025 | 0.2000 | 11 |
| mgc_only | 4.0 | 2 | 0.8354 | 0.7575 | 0.2611 | 11 |
| mgc_only | 4.0 | 3 | 0.8529 | 0.7875 | 0.2250 | 12 |
| mgc_only | 4.0 | 4 | 0.8443 | 0.7700 | 0.2361 | 12 |
| mgc_only | 4.0 | 5 | 0.8149 | 0.7475 | 0.2667 | 11 |
| mgp_mgc | 4.0 | 1 | 0.9128 | 0.8550 | 0.1444 | 11 |
| mgp_mgc | 4.0 | 2 | 0.9219 | 0.8525 | 0.1528 | 11 |
| mgp_mgc | 4.0 | 3 | 0.9278 | 0.8800 | 0.1278 | 12 |
| mgp_mgc | 4.0 | 4 | 0.9213 | 0.8600 | 0.1333 | 12 |
| mgp_mgc | 4.0 | 5 | 0.9078 | 0.8475 | 0.1500 | 11 |
| mgp_only | 0.0 | 1 | 0.8209 | 0.8100 | 0.1889 | 11 |
| mgp_only | 0.0 | 2 | 0.8042 | 0.7675 | 0.2361 | 11 |
| mgp_only | 0.0 | 3 | 0.8367 | 0.8075 | 0.1917 | 12 |
| mgp_only | 0.0 | 4 | 0.8327 | 0.7825 | 0.2167 | 12 |
| mgp_only | 0.0 | 5 | 0.8156 | 0.8050 | 0.1750 | 11 |
| naive_adapter | 0.0 | 1 | 0.7837 | 0.6950 | 0.3250 | 11 |
| naive_adapter | 0.0 | 2 | 0.7719 | 0.6675 | 0.3583 | 11 |
| naive_adapter | 0.0 | 3 | 0.7977 | 0.7225 | 0.2833 | 12 |
| naive_adapter | 0.0 | 4 | 0.7877 | 0.7050 | 0.3056 | 12 |
| naive_adapter | 0.0 | 5 | 0.7606 | 0.6850 | 0.3250 | 11 |
| zero_shot_clap | 0.0 | 1 | 0.8187 | 0.8425 | 0.0389 | 11 |
| zero_shot_clap | 0.0 | 2 | 0.8184 | 0.8200 | 0.0556 | 11 |
| zero_shot_clap | 0.0 | 3 | 0.8197 | 0.8275 | 0.0472 | 12 |
| zero_shot_clap | 0.0 | 4 | 0.8094 | 0.8375 | 0.0306 | 12 |
| zero_shot_clap | 0.0 | 5 | 0.8217 | 0.8275 | 0.0500 | 11 |

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
