# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone msclap --fold all --seed 0 --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 4 --class-order canonical --out-dir /tmp/mgclap_phase2/main_grid/msclap/seed_0`
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
| mgc_only | 4.0 | 1 | 0.7753 | 0.8475 | 0.1583 | 2 |
| mgc_only | 4.0 | 2 | 0.7727 | 0.8400 | 0.1722 | 2 |
| mgc_only | 4.0 | 3 | 0.7713 | 0.8150 | 0.2000 | 2 |
| mgc_only | 4.0 | 4 | 0.7731 | 0.8425 | 0.1611 | 2 |
| mgc_only | 4.0 | 5 | 0.7819 | 0.7650 | 0.2444 | 1 |
| mgp_mgc | 4.0 | 1 | 0.9582 | 0.9400 | 0.0556 | 2 |
| mgp_mgc | 4.0 | 2 | 0.9674 | 0.9650 | 0.0306 | 2 |
| mgp_mgc | 4.0 | 3 | 0.9680 | 0.9125 | 0.0944 | 2 |
| mgp_mgc | 4.0 | 4 | 0.9587 | 0.9625 | 0.0333 | 2 |
| mgp_mgc | 4.0 | 5 | 0.9762 | 0.9625 | 0.0278 | 1 |
| mgp_only | 0.0 | 1 | 0.9002 | 0.9100 | 0.0917 | 2 |
| mgp_only | 0.0 | 2 | 0.9229 | 0.9150 | 0.0861 | 2 |
| mgp_only | 0.0 | 3 | 0.9112 | 0.9025 | 0.1028 | 2 |
| mgp_only | 0.0 | 4 | 0.9095 | 0.9175 | 0.0778 | 2 |
| mgp_only | 0.0 | 5 | 0.9763 | 0.9225 | 0.0611 | 1 |
| naive_adapter | 0.0 | 1 | 0.7314 | 0.8100 | 0.2028 | 2 |
| naive_adapter | 0.0 | 2 | 0.7411 | 0.8025 | 0.2139 | 2 |
| naive_adapter | 0.0 | 3 | 0.7655 | 0.7750 | 0.2472 | 2 |
| naive_adapter | 0.0 | 4 | 0.7602 | 0.8225 | 0.1833 | 2 |
| naive_adapter | 0.0 | 5 | 0.7386 | 0.7675 | 0.2417 | 1 |
| zero_shot_clap | 0.0 | 1 | 0.9377 | 0.9400 | 0.0139 | 2 |
| zero_shot_clap | 0.0 | 2 | 0.9766 | 0.9650 | 0.0111 | 2 |
| zero_shot_clap | 0.0 | 3 | 0.9571 | 0.9325 | 0.0306 | 2 |
| zero_shot_clap | 0.0 | 4 | 0.9472 | 0.9475 | 0.0083 | 2 |
| zero_shot_clap | 0.0 | 5 | 0.9605 | 0.9325 | 0.0250 | 1 |

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
