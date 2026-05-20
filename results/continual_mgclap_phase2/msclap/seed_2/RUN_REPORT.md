# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone msclap --fold all --seed 2 --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order shuffled --out-dir /tmp/mgclap_phase2/main_grid/msclap/seed_2`
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
| mgc_only | 4.0 | 1 | 0.8694 | 0.7900 | 0.2278 | 2 |
| mgc_only | 4.0 | 2 | 0.8690 | 0.7675 | 0.2556 | 2 |
| mgc_only | 4.0 | 3 | 0.8768 | 0.7850 | 0.2361 | 2 |
| mgc_only | 4.0 | 4 | 0.8920 | 0.8175 | 0.2028 | 2 |
| mgc_only | 4.0 | 5 | 0.8741 | 0.7825 | 0.2389 | 2 |
| mgp_mgc | 4.0 | 1 | 0.9834 | 0.9525 | 0.0444 | 2 |
| mgp_mgc | 4.0 | 2 | 0.9865 | 0.9650 | 0.0333 | 2 |
| mgp_mgc | 4.0 | 3 | 0.9784 | 0.9375 | 0.0639 | 2 |
| mgp_mgc | 4.0 | 4 | 0.9854 | 0.9600 | 0.0417 | 2 |
| mgp_mgc | 4.0 | 5 | 0.9717 | 0.9375 | 0.0667 | 2 |
| mgp_only | 0.0 | 1 | 0.9771 | 0.9475 | 0.0472 | 2 |
| mgp_only | 0.0 | 2 | 0.9727 | 0.9400 | 0.0500 | 2 |
| mgp_only | 0.0 | 3 | 0.9743 | 0.9300 | 0.0639 | 2 |
| mgp_only | 0.0 | 4 | 0.9748 | 0.9500 | 0.0500 | 2 |
| mgp_only | 0.0 | 5 | 0.9704 | 0.9275 | 0.0806 | 2 |
| naive_adapter | 0.0 | 1 | 0.8728 | 0.7700 | 0.2472 | 2 |
| naive_adapter | 0.0 | 2 | 0.8745 | 0.7875 | 0.2333 | 2 |
| naive_adapter | 0.0 | 3 | 0.8664 | 0.7575 | 0.2667 | 2 |
| naive_adapter | 0.0 | 4 | 0.8815 | 0.7850 | 0.2389 | 2 |
| naive_adapter | 0.0 | 5 | 0.8479 | 0.7250 | 0.3028 | 2 |
| zero_shot_clap | 0.0 | 1 | 0.9740 | 0.9400 | 0.0278 | 2 |
| zero_shot_clap | 0.0 | 2 | 0.9872 | 0.9650 | 0.0111 | 2 |
| zero_shot_clap | 0.0 | 3 | 0.9731 | 0.9325 | 0.0444 | 2 |
| zero_shot_clap | 0.0 | 4 | 0.9798 | 0.9475 | 0.0306 | 2 |
| zero_shot_clap | 0.0 | 5 | 0.9790 | 0.9325 | 0.0556 | 2 |

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
