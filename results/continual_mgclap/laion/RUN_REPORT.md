# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold all --tasks 10 --classes-per-task 5 --alpha 0.10 --adapter-rank 16 --epochs-max 20 --lr 1e-3 --weight-decay 1e-4 --batch-size 64 --betas 0 1 2 4 8 --seed 0`
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
| mgc_only | 0.0 | 1 | 0.7479 | 0.6550 | 0.3694 | 4 |
| mgc_only | 0.0 | 2 | 0.7037 | 0.5750 | 0.4639 | 4 |
| mgc_only | 0.0 | 3 | 0.7439 | 0.6525 | 0.3694 | 5 |
| mgc_only | 0.0 | 4 | 0.7468 | 0.6525 | 0.3694 | 4 |
| mgc_only | 0.0 | 5 | 0.7043 | 0.6025 | 0.4306 | 4 |
| mgc_only | 1.0 | 1 | 0.7803 | 0.7425 | 0.2722 | 4 |
| mgc_only | 1.0 | 2 | 0.7444 | 0.6575 | 0.3750 | 4 |
| mgc_only | 1.0 | 3 | 0.7898 | 0.7375 | 0.2778 | 5 |
| mgc_only | 1.0 | 4 | 0.7861 | 0.7300 | 0.2833 | 4 |
| mgc_only | 1.0 | 5 | 0.7443 | 0.6650 | 0.3639 | 4 |
| mgc_only | 2.0 | 1 | 0.7845 | 0.7100 | 0.3111 | 4 |
| mgc_only | 2.0 | 2 | 0.7566 | 0.6925 | 0.3306 | 4 |
| mgc_only | 2.0 | 3 | 0.7913 | 0.7775 | 0.2333 | 5 |
| mgc_only | 2.0 | 4 | 0.7735 | 0.7400 | 0.2694 | 4 |
| mgc_only | 2.0 | 5 | 0.7485 | 0.6700 | 0.3528 | 4 |
| mgc_only | 4.0 | 1 | 0.7877 | 0.7475 | 0.2722 | 4 |
| mgc_only | 4.0 | 2 | 0.7684 | 0.6725 | 0.3556 | 4 |
| mgc_only | 4.0 | 3 | 0.7929 | 0.7825 | 0.2278 | 5 |
| mgc_only | 4.0 | 4 | 0.7834 | 0.7375 | 0.2722 | 4 |
| mgc_only | 4.0 | 5 | 0.7367 | 0.6925 | 0.3306 | 4 |
| mgc_only | 8.0 | 1 | 0.7852 | 0.7625 | 0.2500 | 4 |
| mgc_only | 8.0 | 2 | 0.7520 | 0.6925 | 0.3306 | 4 |
| mgc_only | 8.0 | 3 | 0.8053 | 0.7750 | 0.2361 | 5 |
| mgc_only | 8.0 | 4 | 0.7723 | 0.7375 | 0.2750 | 4 |
| mgc_only | 8.0 | 5 | 0.7690 | 0.7275 | 0.2917 | 4 |
| mgp_mgc | 0.0 | 1 | 0.8209 | 0.8150 | 0.1750 | 4 |
| mgp_mgc | 0.0 | 2 | 0.8220 | 0.7875 | 0.2111 | 4 |
| mgp_mgc | 0.0 | 3 | 0.8328 | 0.8100 | 0.1861 | 5 |
| mgp_mgc | 0.0 | 4 | 0.8476 | 0.8225 | 0.1639 | 4 |
| mgp_mgc | 0.0 | 5 | 0.8156 | 0.8225 | 0.1528 | 4 |
| mgp_mgc | 1.0 | 1 | 0.9361 | 0.9100 | 0.0833 | 4 |
| mgp_mgc | 1.0 | 2 | 0.9463 | 0.8925 | 0.1139 | 4 |
| mgp_mgc | 1.0 | 3 | 0.9200 | 0.8975 | 0.1083 | 5 |
| mgp_mgc | 1.0 | 4 | 0.9450 | 0.8950 | 0.0889 | 4 |
| mgp_mgc | 1.0 | 5 | 0.9127 | 0.8825 | 0.1111 | 4 |
| mgp_mgc | 2.0 | 1 | 0.9437 | 0.9225 | 0.0694 | 4 |
| mgp_mgc | 2.0 | 2 | 0.9567 | 0.9100 | 0.0972 | 4 |
| mgp_mgc | 2.0 | 3 | 0.9354 | 0.8850 | 0.1222 | 5 |
| mgp_mgc | 2.0 | 4 | 0.9529 | 0.9050 | 0.0806 | 4 |
| mgp_mgc | 2.0 | 5 | 0.9310 | 0.8850 | 0.1083 | 4 |
| mgp_mgc | 4.0 | 1 | 0.9489 | 0.9275 | 0.0639 | 4 |
| mgp_mgc | 4.0 | 2 | 0.9687 | 0.9125 | 0.0944 | 4 |
| mgp_mgc | 4.0 | 3 | 0.9444 | 0.8825 | 0.1250 | 5 |
| mgp_mgc | 4.0 | 4 | 0.9550 | 0.9100 | 0.0750 | 4 |
| mgp_mgc | 4.0 | 5 | 0.9405 | 0.9100 | 0.0806 | 4 |
| mgp_mgc | 8.0 | 1 | 0.9524 | 0.9225 | 0.0694 | 4 |
| mgp_mgc | 8.0 | 2 | 0.9629 | 0.9175 | 0.0833 | 4 |
| mgp_mgc | 8.0 | 3 | 0.9505 | 0.9025 | 0.1028 | 5 |
| mgp_mgc | 8.0 | 4 | 0.9620 | 0.9200 | 0.0667 | 4 |
| mgp_mgc | 8.0 | 5 | 0.9345 | 0.8850 | 0.1083 | 4 |
| mgp_only | 0.0 | 1 | 0.8241 | 0.8325 | 0.1611 | 4 |
| mgp_only | 0.0 | 2 | 0.8187 | 0.7850 | 0.2111 | 4 |
| mgp_only | 0.0 | 3 | 0.8285 | 0.8125 | 0.1833 | 5 |
| mgp_only | 0.0 | 4 | 0.8493 | 0.8250 | 0.1639 | 4 |
| mgp_only | 0.0 | 5 | 0.8160 | 0.8225 | 0.1556 | 4 |
| naive_adapter | 0.0 | 1 | 0.7286 | 0.6475 | 0.3806 | 4 |
| naive_adapter | 0.0 | 2 | 0.7148 | 0.5925 | 0.4417 | 4 |
| naive_adapter | 0.0 | 3 | 0.7393 | 0.6750 | 0.3500 | 5 |
| naive_adapter | 0.0 | 4 | 0.7441 | 0.6600 | 0.3611 | 4 |
| naive_adapter | 0.0 | 5 | 0.7218 | 0.6300 | 0.3972 | 4 |
| zero_shot_clap | 0.0 | 1 | 0.8187 | 0.8425 | 0.0389 | 4 |
| zero_shot_clap | 0.0 | 2 | 0.8184 | 0.8200 | 0.0556 | 4 |
| zero_shot_clap | 0.0 | 3 | 0.8197 | 0.8275 | 0.0472 | 5 |
| zero_shot_clap | 0.0 | 4 | 0.8094 | 0.8375 | 0.0306 | 4 |
| zero_shot_clap | 0.0 | 5 | 0.8217 | 0.8275 | 0.0500 | 4 |

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
