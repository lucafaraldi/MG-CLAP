# Mind the Gap — CLAP edition

A faithful port of the experiments in **Liang et al., "Mind the Gap: Understanding the
Modality Gap in Multi-modal Contrastive Representation Learning" (NeurIPS 2022,
[arXiv:2203.02053](https://arxiv.org/abs/2203.02053))** from CLIP (image↔text) to
**CLAP** (audio↔text).

Original repo: <https://github.com/Weixin-Liang/Modality-Gap>

## What's being ported

The paper's experiments are organized around its figures and tables. We reproduce each
one with audio replacing images:

| Original | Audio port |
|---|---|
| Fig 1 — gap visualization on CLIP/CLASP/ConVIRT/VideoCLIP (MS-COCO) | Gap visualization on **LAION-CLAP** and **Microsoft CLAP**, evaluated on **AudioCaps** |
| Fig 2 — cone effect at random init (real & noise inputs, MLP layerwise) | Same, with CLAP's audio + text encoders, real audio + Gaussian noise audio |
| Fig 3 — contrastive loss landscape vs gap distance Δ and temperature τ | Synthetic sphere experiment (modality-agnostic) + AudioCaps gap-stat re-derivation |
| Table 1 — zero-shot accuracy under embedding shift (CIFAR/EuroSAT/…) | Zero-shot accuracy on **ESC-50** under shift (5-fold) |
| Table 2 — FairFace/CelebA denigration bias under shift | No clean audio analog — see `table_2_fairness/README.md` for decision |

## Theory cheat sheet

- **L2-normalized embeddings live on the unit hypersphere.**
- **Gap vector:** `Δ = mean(x̂_i) − mean(ŷ_i)` where `x̂, ŷ` are normalized audio and text
  embeddings respectively.
- **Gap distance:** `‖Δ‖₂`. For CLIP on COCO this is ≈ 0.82.
- **Shift intervention** (used in Fig 3 and Table 1):
  ```
  x'_i = normalize(x_i − λ · Δ)
  y'_i = normalize(y_i + λ · Δ)
  ```
  Applied symmetrically to both modalities. λ=0 → identity, λ>0 → close the gap, λ<0 → widen it.
- **InfoNCE / NT-Xent loss** (symmetric): `ℓ = ½(ℓ_{a→t} + ℓ_{t→a})`, with temperature τ
  (CLIP uses τ ≈ 0.01; LAION-CLAP also learns τ).

## Project layout

```
mind_the_gap_CLAP/
├── lib/                       # shared library
│   ├── clap_models.py         # unified LAION-CLAP + MS-CLAP loader
│   ├── datasets.py            # AudioCaps, ESC-50 loaders
│   ├── gap_utils.py           # gap distance, gap direction, shift, NT-Xent
│   └── viz.py                 # PCA / UMAP / plotting helpers
├── scripts/
│   ├── 00_setup_data.py       # download + organize AudioCaps + ESC-50
│   └── 01_extract_embeddings.py  # cache audio+text embeddings (per CLAP backbone)
├── figure_1_modality_gap/     # Fig 1: gap visualization
├── figure_2_cone_effect/      # Fig 2: random-init cone (a, b, c)
├── figure_3_contrastive_learning/  # Fig 3: loss-landscape sphere experiment
├── table_1_zero_shot/         # Table 1: ESC-50 zero-shot vs shift
├── table_2_fairness/          # Table 2: decision doc (no clean audio analog)
├── data/                      # raw datasets (gitignored)
├── embeddings/                # cached .npz embeddings (gitignored)
├── results/                   # figures + numerical results
└── notebooks/                 # optional Jupyter wrappers
```

## Setup

Tested on Python 3.10 (macOS arm64 with MPS, and Ubuntu 22.04 x86_64 / aarch64).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

CLAP models (auto-downloaded on first use):

- **LAION-CLAP** — `laion_clap.CLAP_Module(enable_fusion=False)`, default `630k-audioset-best.pt` checkpoint.
- **Microsoft CLAP** — `msclap.CLAP(version='2023')`.

### Data

```bash
python scripts/00_setup_data.py --audiocaps --esc50
```

- AudioCaps: ~46k audio-caption pairs, requires YouTube audio downloads. The script
  uses the [audiocaps repository](https://github.com/cdjkim/audiocaps) CSVs and
  `yt-dlp` to grab the 10s clips. We default to the **validation split** (~495 pairs)
  to keep this runnable; bump to test/train if you have storage and time.
- ESC-50: ~600 MB, hosted [here](https://github.com/karolpiczak/ESC-50). 50 classes,
  2,000 clips, 5 cross-validation folds.

Both downloads can take time. If you already have these datasets, set
`AUDIOCAPS_ROOT` and `ESC50_ROOT` env vars and skip the script.

## Running the experiments

Each experiment is a runnable Python script (also wrappable in a notebook).

```bash
# 0) one-time: extract and cache embeddings for both CLAP backbones
python scripts/01_extract_embeddings.py --backbone laion --dataset audiocaps --split val
python scripts/01_extract_embeddings.py --backbone msclap --dataset audiocaps --split val
python scripts/01_extract_embeddings.py --backbone laion --dataset esc50
python scripts/01_extract_embeddings.py --backbone msclap --dataset esc50

# Fig 1
python figure_1_modality_gap/run.py --backbone laion
python figure_1_modality_gap/run.py --backbone msclap

# Fig 2
python figure_2_cone_effect/2a_random_init/run.py
python figure_2_cone_effect/2b_random_mlp/run.py
python figure_2_cone_effect/2c_scatter_cones/run.py

# Fig 3
python figure_3_contrastive_learning/run.py

# Table 1 — zero-shot under shift
python table_1_zero_shot/run.py --backbone laion
python table_1_zero_shot/run.py --backbone msclap

# Table 1 — light contrastive training sweep (analog of paper's train_clip.py)
python table_1_zero_shot/training/train_clap.py --backbone laion \
    --temperatures 0.01 0.05 0.1 0.5 1.0 \
    --init-gaps 0.0 0.3 0.6 0.9 --epochs 30

# Figure 2c — pretrained-encoder cone scatter (uses cached embeddings)
python figure_2_cone_effect/2c_scatter_cones/run_pretrained.py
```

A full reproduction audit mapping every notebook in the original repo to this
port lives at [`docs/AUDIT.md`](docs/AUDIT.md).

Outputs land under `results/` (figures + JSON / CSV summary tables) and
`embeddings/` (`.npz` caches).

## Citation

```bibtex
@inproceedings{liang2022mind,
  title     = {Mind the Gap: Understanding the Modality Gap in Multi-modal
               Contrastive Representation Learning},
  author    = {Liang, Weixin and Zhang, Yuhui and Kwon, Yongchan and
               Yeung, Serena and Zou, James},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```
