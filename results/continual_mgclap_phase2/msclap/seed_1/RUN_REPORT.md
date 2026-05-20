# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone msclap --fold all --seed 1 --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order shuffled --out-dir /tmp/mgclap_phase2/main_grid/msclap/seed_1`
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
| mgc_only | 4.0 | 1 | 0.8344 | 0.7600 | 0.2639 | 2 |
| mgc_only | 4.0 | 2 | 0.8188 | 0.7375 | 0.2917 | 2 |
| mgc_only | 4.0 | 3 | 0.8412 | 0.7350 | 0.2944 | 3 |
| mgc_only | 4.0 | 4 | 0.8595 | 0.7400 | 0.2833 | 3 |
| mgc_only | 4.0 | 5 | 0.8371 | 0.7250 | 0.3028 | 3 |
| mgp_mgc | 4.0 | 1 | 0.9632 | 0.9700 | 0.0278 | 2 |
| mgp_mgc | 4.0 | 2 | 0.9738 | 0.9600 | 0.0417 | 2 |
| mgp_mgc | 4.0 | 3 | 0.9510 | 0.9475 | 0.0472 | 3 |
| mgp_mgc | 4.0 | 4 | 0.9744 | 0.9550 | 0.0444 | 3 |
| mgp_mgc | 4.0 | 5 | 0.9519 | 0.9425 | 0.0611 | 3 |
| mgp_only | 0.0 | 1 | 0.9491 | 0.9375 | 0.0639 | 2 |
| mgp_only | 0.0 | 2 | 0.9623 | 0.9525 | 0.0500 | 2 |
| mgp_only | 0.0 | 3 | 0.9335 | 0.9175 | 0.0750 | 3 |
| mgp_only | 0.0 | 4 | 0.9586 | 0.9400 | 0.0611 | 3 |
| mgp_only | 0.0 | 5 | 0.9268 | 0.9400 | 0.0528 | 3 |
| naive_adapter | 0.0 | 1 | 0.8344 | 0.7700 | 0.2528 | 2 |
| naive_adapter | 0.0 | 2 | 0.8235 | 0.7375 | 0.2917 | 2 |
| naive_adapter | 0.0 | 3 | 0.8220 | 0.7175 | 0.3139 | 3 |
| naive_adapter | 0.0 | 4 | 0.8381 | 0.7375 | 0.2889 | 3 |
| naive_adapter | 0.0 | 5 | 0.8313 | 0.7550 | 0.2722 | 3 |
| zero_shot_clap | 0.0 | 1 | 0.9584 | 0.9400 | 0.0389 | 2 |
| zero_shot_clap | 0.0 | 2 | 0.9815 | 0.9650 | 0.0167 | 2 |
| zero_shot_clap | 0.0 | 3 | 0.9593 | 0.9325 | 0.0306 | 3 |
| zero_shot_clap | 0.0 | 4 | 0.9813 | 0.9475 | 0.0222 | 3 |
| zero_shot_clap | 0.0 | 5 | 0.9633 | 0.9325 | 0.0417 | 3 |

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
