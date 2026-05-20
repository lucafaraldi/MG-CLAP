# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone msclap --fold all --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 0 1 2 4 8 --seed 0`
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
| mgc_only | 0.0 | 1 | 0.7496 | 0.8225 | 0.1861 | 2 |
| mgc_only | 0.0 | 2 | 0.7456 | 0.7950 | 0.2222 | 2 |
| mgc_only | 0.0 | 3 | 0.7584 | 0.7825 | 0.2389 | 2 |
| mgc_only | 0.0 | 4 | 0.7582 | 0.7925 | 0.2167 | 1 |
| mgc_only | 0.0 | 5 | 0.7451 | 0.7575 | 0.2528 | 2 |
| mgc_only | 1.0 | 1 | 0.7723 | 0.8250 | 0.1861 | 2 |
| mgc_only | 1.0 | 2 | 0.7735 | 0.8475 | 0.1639 | 2 |
| mgc_only | 1.0 | 3 | 0.7940 | 0.8600 | 0.1528 | 2 |
| mgc_only | 1.0 | 4 | 0.7790 | 0.8350 | 0.1694 | 1 |
| mgc_only | 1.0 | 5 | 0.7653 | 0.8200 | 0.1806 | 2 |
| mgc_only | 2.0 | 1 | 0.7632 | 0.8175 | 0.1917 | 2 |
| mgc_only | 2.0 | 2 | 0.7605 | 0.8075 | 0.2083 | 2 |
| mgc_only | 2.0 | 3 | 0.7786 | 0.8325 | 0.1833 | 2 |
| mgc_only | 2.0 | 4 | 0.7871 | 0.8325 | 0.1722 | 1 |
| mgc_only | 2.0 | 5 | 0.7638 | 0.7850 | 0.2222 | 2 |
| mgc_only | 4.0 | 1 | 0.7606 | 0.8250 | 0.1861 | 2 |
| mgc_only | 4.0 | 2 | 0.7792 | 0.8175 | 0.1972 | 2 |
| mgc_only | 4.0 | 3 | 0.7887 | 0.8275 | 0.1889 | 2 |
| mgc_only | 4.0 | 4 | 0.7874 | 0.8100 | 0.1972 | 1 |
| mgc_only | 4.0 | 5 | 0.7663 | 0.8200 | 0.1806 | 2 |
| mgc_only | 8.0 | 1 | 0.7702 | 0.8350 | 0.1722 | 2 |
| mgc_only | 8.0 | 2 | 0.7676 | 0.8250 | 0.1889 | 2 |
| mgc_only | 8.0 | 3 | 0.7938 | 0.8200 | 0.1944 | 2 |
| mgc_only | 8.0 | 4 | 0.7810 | 0.8225 | 0.1833 | 1 |
| mgc_only | 8.0 | 5 | 0.7612 | 0.7950 | 0.2083 | 2 |
| mgp_mgc | 0.0 | 1 | 0.8963 | 0.9025 | 0.1000 | 2 |
| mgp_mgc | 0.0 | 2 | 0.9254 | 0.9050 | 0.0972 | 2 |
| mgp_mgc | 0.0 | 3 | 0.9107 | 0.8900 | 0.1139 | 2 |
| mgp_mgc | 0.0 | 4 | 0.9524 | 0.9450 | 0.0472 | 1 |
| mgp_mgc | 0.0 | 5 | 0.9274 | 0.8875 | 0.1083 | 2 |
| mgp_mgc | 1.0 | 1 | 0.9479 | 0.9225 | 0.0750 | 2 |
| mgp_mgc | 1.0 | 2 | 0.9598 | 0.9500 | 0.0472 | 2 |
| mgp_mgc | 1.0 | 3 | 0.9628 | 0.8975 | 0.1111 | 2 |
| mgp_mgc | 1.0 | 4 | 0.9854 | 0.9775 | 0.0139 | 1 |
| mgp_mgc | 1.0 | 5 | 0.9611 | 0.9325 | 0.0639 | 2 |
| mgp_mgc | 2.0 | 1 | 0.9525 | 0.9275 | 0.0694 | 2 |
| mgp_mgc | 2.0 | 2 | 0.9729 | 0.9600 | 0.0361 | 2 |
| mgp_mgc | 2.0 | 3 | 0.9660 | 0.8950 | 0.1139 | 2 |
| mgp_mgc | 2.0 | 4 | 0.9876 | 0.9750 | 0.0194 | 1 |
| mgp_mgc | 2.0 | 5 | 0.9657 | 0.9400 | 0.0556 | 2 |
| mgp_mgc | 4.0 | 1 | 0.9609 | 0.9475 | 0.0472 | 2 |
| mgp_mgc | 4.0 | 2 | 0.9691 | 0.9525 | 0.0444 | 2 |
| mgp_mgc | 4.0 | 3 | 0.9606 | 0.9250 | 0.0806 | 2 |
| mgp_mgc | 4.0 | 4 | 0.9873 | 0.9775 | 0.0167 | 1 |
| mgp_mgc | 4.0 | 5 | 0.9659 | 0.9375 | 0.0583 | 2 |
| mgp_mgc | 8.0 | 1 | 0.9582 | 0.9400 | 0.0556 | 2 |
| mgp_mgc | 8.0 | 2 | 0.9695 | 0.9525 | 0.0444 | 2 |
| mgp_mgc | 8.0 | 3 | 0.9665 | 0.9275 | 0.0778 | 2 |
| mgp_mgc | 8.0 | 4 | 0.9885 | 0.9775 | 0.0167 | 1 |
| mgp_mgc | 8.0 | 5 | 0.9640 | 0.9475 | 0.0444 | 2 |
| mgp_only | 0.0 | 1 | 0.9002 | 0.9100 | 0.0917 | 2 |
| mgp_only | 0.0 | 2 | 0.9216 | 0.9075 | 0.0944 | 2 |
| mgp_only | 0.0 | 3 | 0.9102 | 0.8775 | 0.1306 | 2 |
| mgp_only | 0.0 | 4 | 0.9684 | 0.9400 | 0.0528 | 1 |
| mgp_only | 0.0 | 5 | 0.9255 | 0.8975 | 0.0972 | 2 |
| naive_adapter | 0.0 | 1 | 0.7314 | 0.8100 | 0.2028 | 2 |
| naive_adapter | 0.0 | 2 | 0.7535 | 0.7650 | 0.2556 | 2 |
| naive_adapter | 0.0 | 3 | 0.7716 | 0.8100 | 0.2083 | 2 |
| naive_adapter | 0.0 | 4 | 0.7488 | 0.7975 | 0.2111 | 1 |
| naive_adapter | 0.0 | 5 | 0.7418 | 0.7650 | 0.2444 | 2 |
| zero_shot_clap | 0.0 | 1 | 0.9377 | 0.9400 | 0.0139 | 2 |
| zero_shot_clap | 0.0 | 2 | 0.9766 | 0.9650 | 0.0111 | 2 |
| zero_shot_clap | 0.0 | 3 | 0.9571 | 0.9325 | 0.0306 | 2 |
| zero_shot_clap | 0.0 | 4 | 0.9472 | 0.9475 | 0.0083 | 1 |
| zero_shot_clap | 0.0 | 5 | 0.9605 | 0.9325 | 0.0250 | 2 |

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
