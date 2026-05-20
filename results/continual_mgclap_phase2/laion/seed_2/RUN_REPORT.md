# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold all --seed 2 --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order shuffled --out-dir /tmp/mgclap_phase2/main_grid/laion/seed_2`
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
| mgc_only | 4.0 | 1 | 0.8257 | 0.6475 | 0.3861 | 1 |
| mgc_only | 4.0 | 2 | 0.8135 | 0.6125 | 0.4250 | 1 |
| mgc_only | 4.0 | 3 | 0.8400 | 0.6450 | 0.3917 | 1 |
| mgc_only | 4.0 | 4 | 0.8371 | 0.6775 | 0.3528 | 1 |
| mgc_only | 4.0 | 5 | 0.8167 | 0.6550 | 0.3833 | 1 |
| mgp_mgc | 4.0 | 1 | 0.9851 | 0.9700 | 0.0139 | 1 |
| mgp_mgc | 4.0 | 2 | 0.9945 | 0.9875 | 0.0083 | 1 |
| mgp_mgc | 4.0 | 3 | 0.9888 | 0.9725 | 0.0194 | 1 |
| mgp_mgc | 4.0 | 4 | 0.9890 | 0.9675 | 0.0167 | 1 |
| mgp_mgc | 4.0 | 5 | 0.9772 | 0.9550 | 0.0417 | 1 |
| mgp_only | 0.0 | 1 | 0.9202 | 0.8900 | 0.0444 | 1 |
| mgp_only | 0.0 | 2 | 0.9232 | 0.8725 | 0.0694 | 1 |
| mgp_only | 0.0 | 3 | 0.9381 | 0.8775 | 0.0750 | 1 |
| mgp_only | 0.0 | 4 | 0.9164 | 0.8750 | 0.0778 | 1 |
| mgp_only | 0.0 | 5 | 0.9220 | 0.8700 | 0.0889 | 1 |
| naive_adapter | 0.0 | 1 | 0.8195 | 0.6275 | 0.4056 | 1 |
| naive_adapter | 0.0 | 2 | 0.7952 | 0.5950 | 0.4444 | 1 |
| naive_adapter | 0.0 | 3 | 0.7987 | 0.6175 | 0.4222 | 1 |
| naive_adapter | 0.0 | 4 | 0.8162 | 0.6300 | 0.4056 | 1 |
| naive_adapter | 0.0 | 5 | 0.7885 | 0.5925 | 0.4528 | 1 |
| zero_shot_clap | 0.0 | 1 | 0.8802 | 0.8425 | 0.0500 | 1 |
| zero_shot_clap | 0.0 | 2 | 0.8921 | 0.8200 | 0.0611 | 1 |
| zero_shot_clap | 0.0 | 3 | 0.8918 | 0.8275 | 0.0778 | 1 |
| zero_shot_clap | 0.0 | 4 | 0.8904 | 0.8375 | 0.0778 | 1 |
| zero_shot_clap | 0.0 | 5 | 0.8907 | 0.8275 | 0.0833 | 1 |

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
