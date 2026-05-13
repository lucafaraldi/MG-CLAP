# Table 2 — Fairness and bias under embedding shift

## What the original paper does

`Table_2_Implications_CLIP_Fairness/shift_CLIP_FairFace_Bias.ipynb` measures
**denigration bias** in CLIP zero-shot classification on the FairFace dataset:

* 14 FairFace demographic classes (race × gender) + 4 non-human distractors
  (animal, gorilla, chimpanzee, orangutan) + 3 crime-related distractors
  (thief, criminal, suspicious person).
* Baseline CLIP misclassifies a non-trivial fraction of faces — particularly
  faces of certain races — into the offensive distractor classes.
* The headline finding: **increasing the modality gap (negative λ in our shift
  convention) reduces the rate of denigrating misclassifications**, especially
  for underrepresented racial groups.

This is a vision-domain bias study. There is no straightforward audio analog
because faces don't have an obvious audio counterpart that's published with
demographic labels in the same controlled way as FairFace.

## Decision for the CLAP port

The `--scope` choice in our initial planning was **faithful port only — don't go
beyond what's in the paper**. So we explicitly *do not* run a fairness
experiment in this repo.

Three alternative audio-domain bias setups are technically feasible if you
later want to extend; a stub `bias_stub.py` in this folder shows the API.

### Option A — Speaker demographic bias (closest direct port)
* Dataset: **Common Voice** or **VoxCeleb1**, both have gender/age/accent labels.
* Audio: speaker utterances.
* Prompt set: neutral category labels ("person speaking", "voice recording")
  combined with stereotyped or denigrating distractors (occupation prompts,
  socio-economic prompts).
* Metric: per-group misclassification rate into the distractor set, before
  and after embedding shift.
* Effort: moderate — needs careful prompt design to avoid generating new bias.

### Option B — Music genre stereotyping
* Dataset: **GTZAN** or **MagnaTagATune**.
* Audio: music clips across genres.
* Prompt set: genre labels + non-musical socio-cultural distractors.
* Less ethically loaded than Option A, but also further from the paper's intent.

### Option C — Environmental-sound demographic context bias
* Dataset: **VGGSound** with location/scene metadata, or **AudioSet** demographic
  audio events.
* Audio: ambient recordings.
* Prompt set: location/activity prompts + biased distractors.
* Most novel; least directly comparable to the FairFace setup.

## Recommended next step

If the goal is "what does the paper say?", stop here — Table 2's substantive
content is the reduction in face denigration bias under shift, and that finding
doesn't transplant to audio without choosing one of the options above.

If you do want to pursue Option A (speaker demographics), the rest of this repo
is already set up for it: extract per-group audio embeddings the same way as
ESC-50 (`scripts/01_extract_embeddings.py`), then call `lib.gap_utils.shift_features`
exactly as `table_1_zero_shot/run.py` does — only the labels and scoring change.

The `bias_stub.py` in this folder sketches the loop.
