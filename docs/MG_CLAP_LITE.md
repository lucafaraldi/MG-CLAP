# MG-CLAP-lite

## Why MG-CLAP-lite Instead of Full CLAP Fine-Tuning

Full CLAP continual fine-tuning is expensive, slow, and hard to iterate on in a
course setting. MG-CLAP-lite freezes the pretrained CLAP embeddings and asks a
more targeted question:

Can the two MG-CLIP ideas still help when we only adapt the audio side with a
small residual adapter and evaluate class-incremental ESC-50?

This keeps the experiment cheap enough to run while preserving the key geometry
that the modality-gap story depends on.

## What Is Preserved

MG-CLIP's preservation idea is translated into a feature-level stopping rule.

- We track the mean cosine similarity between adapted audio features and
  non-matching text embeddings on task 1.
- We measure relative drift from the initial value.
- We pick `e_star` as the last epoch before the drift exceeds `alpha=10%`.
- Later tasks use exactly `e_star` epochs.

The intent is to stop the adapter before it destroys cross-modal geometry.

## What Is Compensated

MG-CLIP's compensation idea is translated into an audio-space prototype head.

- The text classifier remains the usual CLAP text-embedding classifier.
- After each task, we compute one audio prototype per newly learned class.
- Inference combines text logits and prototype logits with a scalar `beta`.

This gives the model an intra-modal route to recover class information that the
text prompts alone may not express well.

## What Differs From Original MG-CLIP

- No end-to-end CLAP fine-tuning.
- No image branch; audio embeddings only.
- No replay memory.
- The preservation signal is negative audio-text similarity drift rather than
  the exact original CLIP training protocol.
- Compensation uses frozen-audio prototypes on adapted features rather than a
  separately trained visual classifier.

## Why ESC-50

ESC-50 is a practical class-incremental benchmark for this repo because:

- it already has cached CLAP embeddings in the project design,
- it has official five-fold splits,
- it is small enough to run repeatedly, and
- it supports a clean 50-class, 10-task, 5-classes-per-task setup.

## Limitations

- This is a frozen-embedding adapter study, not proof that full CLAP continual
  fine-tuning would behave the same way.
- Beta sweeps are exploratory unless you add a separate validation protocol.
- Results depend on the quality of the underlying CLAP text prompts.
- No replay means naive forgetting can still be severe.

## How To Run

Check what is already present:

```bash
ls -R embeddings data results | head -300
```

If caches are missing:

```bash
python scripts/01_extract_embeddings.py --backbone laion --dataset audiocaps --split val
python scripts/01_extract_embeddings.py --backbone laion --dataset esc50
```

Optional MSCLAP caches:

```bash
python scripts/01_extract_embeddings.py --backbone msclap --dataset audiocaps --split val
python scripts/01_extract_embeddings.py --backbone msclap --dataset esc50
```

Dry run:

```bash
python table_1_zero_shot/continual_mgclap.py --backbone laion --fold 1 --dry-run
```

Full LAION run:

```bash
python table_1_zero_shot/continual_mgclap.py \
  --backbone laion \
  --fold all \
  --tasks 10 \
  --classes-per-task 5 \
  --alpha 0.10 \
  --adapter-rank 16 \
  --epochs-max 20 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --batch-size 64 \
  --betas 0 1 2 4 8 \
  --seed 0
```

## How To Interpret Outcomes

Positive result:

- `mgp_only` beats `naive_adapter` or shows smaller negative-similarity drift
  with similar accuracy.
- `mgc_only` or `mgp_mgc` beats text-only continual baselines.
- `mgp_mgc` gives the best average or final seen-class accuracy.

Null result:

- Preservation does not improve accuracy over naive adaptation.
- Prototype compensation gives no stable gain across folds.

Negative result:

- Preservation hurts learning by stopping too early.
- Prototype compensation amplifies confusion and lowers final accuracy.
- Combined `mgp_mgc` underperforms both simpler baselines.
