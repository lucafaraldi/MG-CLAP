# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold all --seed 0 --tasks 10 --classes-per-task 5 --alpha 0.05 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order canonical --out-dir /tmp/mgclap_phase2/alpha_rank/alpha_0p05__rank_16`
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
| mgc_only | 4.0 | 1 | 0.7961 | 0.7275 | 0.2917 | 4 |
| mgc_only | 4.0 | 2 | 0.7578 | 0.6925 | 0.3333 | 4 |
| mgc_only | 4.0 | 3 | 0.7975 | 0.7550 | 0.2611 | 4 |
| mgc_only | 4.0 | 4 | 0.7793 | 0.7075 | 0.3056 | 4 |
| mgc_only | 4.0 | 5 | 0.7347 | 0.6625 | 0.3639 | 3 |
| mgp_mgc | 4.0 | 1 | 0.9506 | 0.9250 | 0.0667 | 4 |
| mgp_mgc | 4.0 | 2 | 0.9677 | 0.9150 | 0.0917 | 4 |
| mgp_mgc | 4.0 | 3 | 0.9670 | 0.9325 | 0.0694 | 4 |
| mgp_mgc | 4.0 | 4 | 0.9574 | 0.9175 | 0.0639 | 4 |
| mgp_mgc | 4.0 | 5 | 0.9552 | 0.9375 | 0.0528 | 3 |
| mgp_only | 0.0 | 1 | 0.8241 | 0.8325 | 0.1611 | 4 |
| mgp_only | 0.0 | 2 | 0.8161 | 0.7875 | 0.2083 | 4 |
| mgp_only | 0.0 | 3 | 0.8499 | 0.8225 | 0.1694 | 4 |
| mgp_only | 0.0 | 4 | 0.8376 | 0.8050 | 0.1806 | 4 |
| mgp_only | 0.0 | 5 | 0.8321 | 0.8200 | 0.1556 | 3 |
| naive_adapter | 0.0 | 1 | 0.7286 | 0.6475 | 0.3806 | 4 |
| naive_adapter | 0.0 | 2 | 0.7158 | 0.5775 | 0.4583 | 4 |
| naive_adapter | 0.0 | 3 | 0.7410 | 0.6600 | 0.3639 | 4 |
| naive_adapter | 0.0 | 4 | 0.7480 | 0.6800 | 0.3389 | 4 |
| naive_adapter | 0.0 | 5 | 0.7007 | 0.5600 | 0.4778 | 3 |
| zero_shot_clap | 0.0 | 1 | 0.8187 | 0.8425 | 0.0389 | 4 |
| zero_shot_clap | 0.0 | 2 | 0.8184 | 0.8200 | 0.0556 | 4 |
| zero_shot_clap | 0.0 | 3 | 0.8197 | 0.8275 | 0.0472 | 4 |
| zero_shot_clap | 0.0 | 4 | 0.8094 | 0.8375 | 0.0306 | 4 |
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
