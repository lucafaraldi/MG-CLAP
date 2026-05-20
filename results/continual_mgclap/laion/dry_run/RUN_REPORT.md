# MG-CLAP-lite Run Report

## What Was Run

- Command: `table_1_zero_shot/continual_mgclap.py --backbone laion --fold 1 --dry-run`
- Backbone: `laion`
- Folds: `[1]`
- Tasks x classes/task: `10 x 5`
- Dry run: `True`

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
| mgc_only | 0.0 | 1 | 0.9062 | 0.9125 | 0.0750 | 2 |
| mgc_only | 1.0 | 1 | 1.0000 | 1.0000 | 0.0000 | 2 |
| mgc_only | 2.0 | 1 | 1.0000 | 1.0000 | 0.0000 | 2 |
| mgc_only | 4.0 | 1 | 1.0000 | 1.0000 | 0.0000 | 2 |
| mgc_only | 8.0 | 1 | 0.9938 | 0.9875 | 0.0000 | 2 |
| mgp_mgc | 0.0 | 1 | 0.9125 | 0.9250 | 0.0500 | 2 |
| mgp_mgc | 1.0 | 1 | 1.0000 | 1.0000 | 0.0000 | 2 |
| mgp_mgc | 2.0 | 1 | 0.9938 | 0.9875 | 0.0000 | 2 |
| mgp_mgc | 4.0 | 1 | 0.9938 | 0.9875 | 0.0000 | 2 |
| mgp_mgc | 8.0 | 1 | 0.9938 | 0.9875 | 0.0000 | 2 |
| mgp_only | 0.0 | 1 | 0.9125 | 0.9250 | 0.0500 | 2 |
| naive_adapter | 0.0 | 1 | 0.9062 | 0.9125 | 0.0500 | 2 |
| zero_shot_clap | 0.0 | 1 | 0.7937 | 0.7875 | 0.0250 | 2 |

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
