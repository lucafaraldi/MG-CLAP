# Scientific Report Guide for the MG-CLAP Results

This document explains how to turn the current MG-CLAP result set into a scientific report. It is written as a bridge between the code/results in this repository and the final prose of a paper or thesis-style report.

The safest framing is:

> We study whether the modality-gap analysis and preserve-plus-compensate continual-learning idea from MG-CLIP transfer from image-text CLIP to audio-text CLAP. Our work does not claim full CLAP fine-tuning. It is a controlled frozen-embedding study using real AudioCaps and ESC-50 CLAP embeddings, with a lightweight audio-side adapter for continual learning.

## 1. Core Premise

Contrastive audio-text models such as CLAP embed audio clips and text prompts into a shared vector space. Zero-shot classification then works by comparing an audio embedding with candidate text-label embeddings and choosing the most similar label.

The original "Mind the Gap" idea in CLIP is that the two modalities may not occupy the same region of the shared embedding space. Even when paired image/text or audio/text embeddings are close enough for retrieval or classification, the global audio and text clouds can have different centroids. This offset is called the modality gap.

For this project, define normalized audio embeddings as `a_i` and normalized text embeddings as `t_i`. The modality-gap vector is:

```text
Delta = mean(a_i) - mean(t_i)
```

The gap distance is:

```text
||Delta||_2
```

The key scientific question is not only whether a gap exists. The larger question is whether the gap is functionally meaningful. If contrastive learning creates or preserves this geometry, then downstream adaptation that damages the geometry may hurt zero-shot transfer. Conversely, a method that preserves cross-modal geometry while adding task-specific discrimination may work better in continual learning.

## 2. High-Level Research Questions

The report can be organized around four questions:

1. Does CLAP show a measurable audio-text modality gap on real data?
2. Does contrastive-loss behavior support the idea that the gap is tied to training geometry?
3. Does manipulating the gap affect zero-shot audio classification?
4. In continual ESC-50 learning, does a preserve-plus-compensate strategy outperform naive adaptation?

These questions move from descriptive geometry, to mechanism, to downstream behavior, to a practical continual-learning method.

## 3. Evidence Scope and Presentation Safety

The main report should use the real-data results marked safe in `results/PROJECT_STATUS.md`:

| Evidence block | Dataset | Backbones | Safe main claim? | Notes |
|---|---|---|---|---|
| Figure 1 modality gap | AudioCaps validation | LAION-CLAP, MSCLAP | Yes | Real cached audio-text embeddings. |
| Figure 3 loss landscape | AudioCaps validation | LAION-CLAP, MSCLAP | Yes | Real cached embeddings and controlled gap shift. |
| Table 1 shift sweep | ESC-50 | LAION-CLAP, MSCLAP | Yes | Real ESC-50 zero-shot shift sweep. |
| MG-CLAP-lite continual learning | ESC-50 | LAION-CLAP, MSCLAP | Yes | Frozen-embedding adapter study. |
| Phase 2 robustness | ESC-50 | LAION-CLAP, MSCLAP | Yes | Seed/order checks and LAION ablations. |
| Figure 2 cone effect | Synthetic/random-init probes | Synthetic/random models | Supporting only | Useful for motivation, not a real-data headline claim. |
| Synthetic Figure 3 and synthetic Table 1 | Synthetic | N/A | No | Do not use as main empirical evidence. |
| Table 1 training sweep | Mixed/unclear provenance | LAION | No | Mention only as exploratory or omit. |

The report should be explicit that the strongest empirical conclusions come from real AudioCaps and ESC-50 cached embeddings.

## 4. Suggested Report Structure

Use the following structure for a scientific report:

1. **Introduction**
   State that multimodal contrastive models rely on a shared embedding space, but shared does not necessarily mean geometrically identical across modalities. Introduce the modality gap and why it matters for zero-shot classification and continual learning.

2. **Background**
   Summarize CLAP zero-shot classification, the original MG-CLIP premise, and the preserve-plus-compensate idea. Keep this section conceptual and define the gap vector.

3. **Method**
   Describe the datasets, backbones, cached embeddings, shift intervention, and MG-CLAP-lite adapter. Make clear that encoders are frozen in the continual-learning study.

4. **Experiments**
   Present experiments in this order:
   - real AudioCaps modality-gap measurement;
   - Figure 3 contrastive-loss landscape under gap shift;
   - ESC-50 zero-shot shift sweep;
   - MG-CLAP-lite continual learning;
   - Phase 2 robustness and ablations;
   - supporting synthetic/random cone-effect diagnostics.

5. **Discussion**
   Explain the full scientific picture: CLAP has a gap, the loss landscape makes the gap plausible as training geometry, gap shifting has backbone-dependent effects, and preservation plus compensation is useful for continual adaptation when judged mainly by average and final seen-class accuracy.

6. **Limitations**
   State that this is not full CLAP fine-tuning; beta is exploratory unless separately validation-tuned; only ESC-50 and AudioCaps are used; no replay memory is used; Figure 2 is not real-data validated.

7. **Conclusion**
   The safe final claim is that the MG-CLIP principle transfers to CLAP at the frozen-embedding adapter level, especially for continual ESC-50 learning.

## 5. Experiment 1: Measuring the CLAP Modality Gap

### Why this experiment was done

Before testing any method, we need to establish that the object of study exists in CLAP. The original MG-CLIP work studied image-text embedding geometry. This project asks whether an analogous audio-text gap appears in CLAP. If CLAP audio and text embeddings have no meaningful gap, then the rest of the MG-style transfer story would be weak.

### What was done

The experiment used real AudioCaps validation embeddings from two CLAP backbones:

- LAION-CLAP with 512-dimensional embeddings.
- Microsoft CLAP with 1024-dimensional embeddings.

For each backbone, audio and text embeddings were L2-normalized, modality centroids were computed, and the distance between centroids was reported as the modality gap.

### Main results

| Backbone | Pairs | Dim | Gap distance | Audio centroid norm | Text centroid norm | Pair cosine mean | Audio cone mean | Text cone mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LAION-CLAP | 1290 | 512 | 0.4480 | 0.3694 | 0.3516 | 0.4962 | 0.1358 | 0.1229 |
| MSCLAP | 1290 | 1024 | 0.7786 | 0.5492 | 0.6304 | 0.3386 | 0.3011 | 0.3970 |

### What the outcome means

Both CLAP backbones show a non-zero audio-text modality gap on real AudioCaps embeddings. MSCLAP has a substantially larger gap than LAION-CLAP. MSCLAP also has larger cone statistics, suggesting stronger modality-wise concentration or anisotropy.

The report should interpret this as evidence that modality-gap geometry is not unique to image-text CLIP. It appears in audio-text CLAP as well. However, the size and structure of the gap are backbone-dependent.

### Suggested report wording

> On AudioCaps, both CLAP backbones exhibit a measurable audio-text modality gap. The gap is smaller for LAION-CLAP and much larger for MSCLAP, indicating that audio-text alignment does not eliminate global modality separation. This motivates the later experiments: if the modalities occupy offset regions, then downstream adaptation may need to preserve this structure rather than collapse it indiscriminately.

## 6. Experiment 2: Cone-Effect Diagnostics

### Why this experiment was done

The original modality-gap literature argues that neural network representations often form cones: embeddings are not uniformly spread over the hypersphere but concentrate around preferred directions. This can help explain why two modalities occupy separated but internally concentrated regions.

The Figure 2 experiments ask whether cone-like structure can arise even before full semantic training, especially through random networks and nonlinear activations. This is a mechanistic diagnostic, not a final downstream result.

### What was done

The repository contains three supporting probes:

- `2a_random_init`: compares real-like and noise inputs through random initializations.
- `2b_random_mlp`: studies layer-wise cone formation across activations such as linear, ReLU, GELU, and sigmoid.
- `2c_scatter_cones`: compares cone and pairwise-gap statistics across random encoders.

### Main results

For `2a_random_init`, the aggregate multi-seed results show similar cone and gap behavior for real-like and noise inputs:

| Input | Audio cone mean | Text cone mean | Gap distance mean |
|---|---:|---:|---:|
| Real-like | 0.6195 | 0.6129 | 0.7675 |
| Noise | 0.6274 | 0.6232 | 0.7797 |

For `2b_random_mlp`, nonlinear activations produce strong cone growth across layers. ReLU and GELU rise toward very high cone values, and sigmoid saturates near 1.0 in deeper layers. Linear layers show weaker cone formation.

For `2c_scatter_cones`, four random encoders had per-encoder cone means around 0.59 to 0.63, with pairwise gap mean about 0.789.

### What the outcome means

These results support the idea that cone structure can emerge from architecture and activation geometry, not only from semantic supervision. This helps explain why modality gaps can be structural. However, because these are synthetic/random-init diagnostics, they should not be used as the main evidence that pretrained CLAP has a real modality gap. That main evidence comes from Experiment 1.

### How to report it

Report Figure 2 as a supporting mechanism:

> Random-initialization and random-MLP probes show that cone-like embedding concentration can arise without semantic training, especially under nonlinear activations. This supports the plausibility of a structural geometric component behind modality gaps. We use this as mechanistic context rather than as primary evidence for pretrained CLAP behavior.

## 7. Experiment 3: Contrastive Loss Landscape Under Gap Shift

### Why this experiment was done

After measuring a gap, the next question is whether the gap is merely an incidental statistic or whether it interacts with the contrastive objective. The original MG paper argued that contrastive learning and temperature can make non-zero gaps loss-preferred. This experiment tests whether an analogous effect appears in real CLAP embeddings.

### What was done

Using real AudioCaps embeddings, the experiment shifted audio and text embeddings along the measured gap direction:

```text
a'_i = normalize(a_i - lambda * Delta)
t'_i = normalize(t_i + lambda * Delta)
```

Here, `lambda = 0` leaves embeddings unchanged. Positive lambda tends to close the gap, while negative lambda tends to widen it. For each shifted embedding set, the symmetric InfoNCE loss was computed at several temperatures.

### Main gap statistics

| Backbone | Pairs | Dim | Gap distance |
|---|---:|---:|---:|
| LAION-CLAP | 590 | 512 | 0.4600 |
| MSCLAP | 590 | 1024 | 0.7652 |

These numbers are close to the Figure 1 pattern: MSCLAP has a larger audio-text gap than LAION-CLAP.

### Temperature-dependent minima

The saved Figure 3 minima table shows that the loss-preferred gap changes with temperature:

| Backbone | Temperature | Best lambda | Gap at minimum loss |
|---|---:|---:|---:|
| LAION-CLAP | 0.01 | -0.50 | 0.8524 |
| LAION-CLAP | 0.02 | -0.50 | 0.8524 |
| LAION-CLAP | 0.0333 | -0.20 | 0.6271 |
| LAION-CLAP | 0.05 | 0.15 | 0.3274 |
| LAION-CLAP | 0.10 | 0.45 | 0.0507 |
| LAION-CLAP | 1.00 | 0.50 | 0.0072 |
| MSCLAP | 0.01 | 1.50 | 1.2752 |
| MSCLAP | 0.02 | 1.50 | 1.2752 |
| MSCLAP | 0.0333 | 0.50 | 0.0231 |
| MSCLAP | 0.05 | 0.50 | 0.0231 |
| MSCLAP | 0.10 | 0.50 | 0.0231 |
| MSCLAP | 1.00 | 0.50 | 0.0231 |

### What the outcome means

The central finding is that the loss-preferred gap is temperature-dependent. For LAION-CLAP, low temperatures prefer a larger non-zero gap, while higher temperatures move the minimum toward a near-zero gap. MSCLAP also shows temperature-sensitive behavior, though its landscape is more backbone-specific.

This matters because contrastive temperature controls how sharply positives and negatives are separated. At low temperature, the loss can reward geometries that separate modalities or neighborhoods more strongly. At higher temperature, the pressure is smoother and can favor smaller gaps.

### What to avoid claiming

Do not claim that this proves actual CLAP training will converge to exactly these gaps. The experiment is a controlled post-hoc landscape probe on cached embeddings. It shows that the contrastive objective is sensitive to gap manipulation, not that the shift sweep is a full training simulation.

### Suggested report wording

> The contrastive-loss landscape shows that gap size is not an inert descriptive statistic. When embeddings are shifted along the measured gap direction, the InfoNCE loss changes systematically, and the loss-minimizing gap depends on temperature. This supports the mechanistic claim that modality-gap geometry is tied to the contrastive objective.

## 8. Experiment 4: ESC-50 Zero-Shot Shift Sweep

### Why this experiment was done

The previous experiments established that a gap exists and that contrastive loss responds to gap shifts. The next question is whether manipulating the gap changes downstream zero-shot classification. ESC-50 is used because it is a standard 50-class environmental sound dataset with official five-fold splits and cached CLAP embeddings in this project.

### What was done

The experiment measured zero-shot ESC-50 accuracy under shifts along the AudioCaps-derived gap direction. The shift sweep used lambdas from -1.0 to 1.0. Accuracy was reported across the five ESC-50 folds.

### Main results

| Backbone | Baseline mean accuracy | Best shifted accuracy | Best shift interpretation |
|---|---:|---:|---|
| LAION-CLAP | 0.8310 | 0.8550 | Clear gain from gap manipulation. |
| MSCLAP | 0.9435 | 0.9440 | Almost no gain; baseline nearly saturated. |

For LAION-CLAP, widening the gap with negative lambda reduced performance, while moderate positive shifts improved performance. The best LAION shifted accuracy is about 2.4 percentage points above baseline.

For MSCLAP, the baseline is already very high. Shifted accuracy stays essentially flat, with only a 0.05 percentage point gain at best.

### What the outcome means

This experiment shows that the downstream effect of gap manipulation is backbone-dependent.

For LAION-CLAP, the gap direction is behaviorally meaningful: changing it can improve or hurt ESC-50 zero-shot classification. For MSCLAP, the model is already strong enough that this particular intervention has little practical effect.

The scientific interpretation should be careful:

- The gap matters for downstream behavior in at least one real CLAP backbone.
- The effect is not universal in magnitude.
- Stronger or better-calibrated backbones may have less room for simple shift-based gains.

### Suggested report wording

> Gap shifting improves LAION-CLAP zero-shot ESC-50 accuracy from 0.8310 to 0.8550, but has almost no effect on MSCLAP, whose baseline is already 0.9435. This indicates that modality-gap manipulation can be behaviorally relevant, but the practical benefit depends on the pretrained backbone and baseline saturation.

## 9. Experiment 5: MG-CLAP-Lite Continual Learning

### Why this experiment was done

The main extension of the project is not just to observe a gap, but to test whether MG-CLIP's preserve-plus-compensate principle helps in audio continual learning.

Continual learning is challenging because sequential adaptation can improve new classes while damaging old-class performance or cross-modal alignment. In CLAP, this is especially risky because zero-shot classification depends on the geometry between audio embeddings and text-label embeddings. If an adapter moves audio features away from the text space, classification can degrade.

### What was done

The experiment uses ESC-50 as a 10-task class-incremental benchmark:

- 50 classes.
- 10 tasks.
- 5 classes per task.
- Evaluation after each task is over all seen classes.
- CLAP encoders are frozen.
- A low-rank residual adapter is trained on audio embeddings.
- Text embeddings remain fixed.

The methods are:

| Method | Meaning |
|---|---|
| `zero_shot_clap` | No adaptation; classify with frozen CLAP audio-text similarity. |
| `naive_adapter` | Sequentially train the audio adapter without preservation or compensation. |
| `mgp_only` | Modality-gap preservation through early stopping based on negative audio-text drift. |
| `mgc_only` | Modality-gap compensation through audio-space prototypes. |
| `mgp_mgc` | Combined preservation and compensation. |

### Preservation rule

The preservation rule monitors negative audio-text similarity drift on task 1. It selects `e_star` as the last epoch before relative drift exceeds `alpha`; if the drift already exceeds `alpha` at epoch 1, the implementation selects epoch 1 rather than epoch 0. The selected task-1 adapter state initializes the preservation-based methods, and later tasks use `e_star` epochs. The intent is to stop training before the adapter destroys cross-modal geometry.

### Compensation rule

The compensation rule builds audio prototypes for seen classes and combines text-classifier logits with prototype logits:

```text
score = text_logit + beta * prototype_logit
```

The beta sweep is exploratory in the original continual results. Phase 2 fixes `beta = 4` for robustness checks.

### Single-run continual results

The first real continual run shows large gains for the combined method:

| Backbone | Method | Avg | Last | Forgetting proxy |
|---|---|---:|---:|---:|
| LAION-CLAP | zero_shot_clap | 0.8176 | 0.8310 | 0.0444 |
| LAION-CLAP | naive_adapter | 0.7297 | 0.6410 | 0.3861 |
| LAION-CLAP | mgp_only | 0.8273 | 0.8155 | 0.1750 |
| LAION-CLAP | best mgp_mgc | 0.9524 | 0.9095 | 0.0861 |
| MSCLAP | zero_shot_clap | 0.9558 | 0.9435 | 0.0178 |
| MSCLAP | naive_adapter | 0.7494 | 0.7895 | 0.2244 |
| MSCLAP | mgp_only | 0.9252 | 0.9065 | 0.0933 |
| MSCLAP | best mgp_mgc | 0.9693 | 0.9490 | 0.0478 |

### What the outcome means

The most important result is that naive sequential adaptation is harmful. It severely reduces average and final accuracy on both backbones. This confirms the risk that adapting audio embeddings without protecting cross-modal geometry can damage CLAP's useful pretrained structure.

The combined method `mgp_mgc` performs best by average and final seen-class accuracy. For LAION-CLAP, it gives a large improvement over both zero-shot CLAP and the naive adapter. For MSCLAP, the improvement over zero-shot is smaller because zero-shot MSCLAP is already very strong, but the combined method still improves average and final accuracy.

Preservation alone is mixed. It helps compared with naive adaptation but does not always beat the zero-shot baseline, especially for MSCLAP. Compensation alone is also not enough. The combination is the clearest positive result.

### Suggested report wording

> In class-incremental ESC-50, naive audio-side adaptation is consistently harmful, indicating that the pretrained audio-text geometry is easy to disrupt. The combined MG-CLAP-lite method, which preserves cross-modal geometry and adds audio-space prototype compensation, gives the strongest average and final seen-class accuracy on both CLAP backbones. This supports the transfer of the MG-CLIP preserve-plus-compensate principle to audio-text CLAP at the frozen-embedding adapter level.

## 10. Experiment 6: Phase 2 Robustness Checks

### Why these experiments were done

The original continual results could be criticized as depending on a single class order, seed, or beta choice. Phase 2 was run to make the conclusion more robust:

- Use fixed `beta = 4` instead of reporting only best beta.
- Run multiple seeds and shuffled class orders.
- Test preservation threshold `alpha`.
- Test adapter rank.

This makes the report's claims more credible.

### Fixed-beta main table

With `beta = 4`, four runs per backbone were found. The combined method remains the strongest by average and final seen-class accuracy.

| Backbone | Method | Avg | Last | Forgetting proxy |
|---|---|---:|---:|---:|
| LAION-CLAP | zero_shot_clap | 0.8685 +/- 0.0303 | 0.8310 +/- 0.0000 | 0.0565 +/- 0.0166 |
| LAION-CLAP | naive_adapter | 0.7721 +/- 0.0307 | 0.6629 +/- 0.0604 | 0.3685 +/- 0.0655 |
| LAION-CLAP | mgp_only | 0.8994 +/- 0.0423 | 0.8683 +/- 0.0370 | 0.0993 +/- 0.0511 |
| LAION-CLAP | mgc_only | 0.8104 +/- 0.0194 | 0.7095 +/- 0.0573 | 0.3157 +/- 0.0639 |
| LAION-CLAP | mgp_mgc | 0.9694 +/- 0.0156 | 0.9417 +/- 0.0285 | 0.0511 +/- 0.0309 |
| MSCLAP | zero_shot_clap | 0.9685 +/- 0.0082 | 0.9435 +/- 0.0000 | 0.0274 +/- 0.0059 |
| MSCLAP | naive_adapter | 0.8188 +/- 0.0442 | 0.7720 +/- 0.0197 | 0.2482 +/- 0.0251 |
| MSCLAP | mgp_only | 0.9545 +/- 0.0209 | 0.9365 +/- 0.0151 | 0.0597 +/- 0.0169 |
| MSCLAP | mgc_only | 0.8388 +/- 0.0394 | 0.7924 +/- 0.0333 | 0.2251 +/- 0.0397 |
| MSCLAP | mgp_mgc | 0.9731 +/- 0.0089 | 0.9570 +/- 0.0101 | 0.0404 +/- 0.0126 |

### What the fixed-beta table means

The combined method is not only best when beta is chosen from an exploratory sweep. It remains best by average and final seen-class accuracy under a fixed beta value across seeds and class orders. This is one of the strongest pieces of evidence in the project.

For LAION-CLAP, `mgp_mgc` improves strongly over zero-shot and dramatically over naive adaptation. For MSCLAP, the gain over zero-shot is small but consistent in average and final accuracy, and the method avoids the collapse seen in naive adaptation. Forgetting should be interpreted separately: for MSCLAP at fixed `beta = 4`, `zero_shot_clap` has a lower forgetting proxy than `mgp_mgc`, even though `mgp_mgc` has higher average and final accuracy.

### Class-order robustness

The shuffled class-order results show that `mgp_mgc` remains strong:

| Backbone | Method | Canonical Avg | Shuffled Avg | Canonical Last | Shuffled Last |
|---|---|---:|---:|---:|---:|
| LAION-CLAP | mgp_mgc | 0.9453 | 0.9774 +/- 0.0080 | 0.9040 | 0.9543 +/- 0.0211 |
| MSCLAP | mgp_mgc | 0.9657 | 0.9756 +/- 0.0090 | 0.9485 | 0.9598 +/- 0.0102 |

The report should acknowledge that some shuffled orders are easier than the canonical order. The correct interpretation is not that order does not matter. The correct interpretation is that the combined method remains strong across the tested orders.

### Alpha sensitivity

The LAION alpha ablation is a marginal summary from the 4x4 alpha-rank grid: each alpha value below is averaged over adapter ranks 4, 8, 16, and 32. Under that marginal summary, stricter preservation is better for `mgp_mgc`:

| Alpha | Avg | Last | Forgetting proxy |
|---:|---:|---:|---:|
| 0.05 | 0.9589 | 0.9226 | 0.0711 |
| 0.10 | 0.9482 | 0.9054 | 0.0899 |
| 0.20 | 0.9131 | 0.8575 | 0.1431 |
| 0.30 | 0.8813 | 0.8256 | 0.1783 |

This supports the preservation hypothesis within the tested LAION grid. Allowing more negative-similarity drift leads to lower performance and more forgetting after averaging across ranks. In report language, smaller `alpha` means stricter geometry preservation, and stricter preservation works better in this setting.

### Adapter-rank sensitivity

The LAION rank ablation is also a marginal summary from the same 4x4 grid: each rank value below is averaged over `alpha` values 0.05, 0.10, 0.20, and 0.30. For LAION `mgp_mgc`, rank sensitivity is mild:

| Rank | Avg | Last | Forgetting proxy |
|---:|---:|---:|---:|
| 4 | 0.9282 | 0.8781 | 0.1197 |
| 8 | 0.9300 | 0.8834 | 0.1149 |
| 16 | 0.9223 | 0.8745 | 0.1247 |
| 32 | 0.9210 | 0.8751 | 0.1231 |

The combined method does not require a high-rank adapter to work in this marginal summary. Smaller ranks are at least competitive in the tested range. This is useful because it suggests the method's benefit is not simply due to adding a large number of trainable parameters.

## 11. Integrated Scientific Interpretation

The whole project supports a layered argument:

1. **CLAP has a modality gap.** Real AudioCaps measurements show non-zero audio-text centroid separation for both LAION-CLAP and MSCLAP.

2. **The gap is connected to contrastive geometry.** Loss-landscape probes show that InfoNCE loss changes under gap shifts and that the preferred gap depends on temperature.

3. **The gap can matter downstream.** ESC-50 shift sweeps show clear behavior changes for LAION-CLAP, although MSCLAP is close to saturated and shows almost no gain.

4. **Naive adaptation is dangerous.** In continual ESC-50 learning, the naive adapter damages performance badly, consistent with the idea that cross-modal geometry is fragile.

5. **Preserve plus compensate is the strongest practical principle for accuracy.** Preservation alone stabilizes but can underfit. Compensation alone helps but does not protect geometry. The combined method gives the best average and final continual-learning accuracy, while forgetting metrics should be discussed separately.

The larger inference is that pretrained multimodal models should not be adapted as if their embedding spaces were ordinary unimodal feature spaces. Their cross-modal geometry is part of the model's useful knowledge. Continual-learning methods for CLAP should therefore balance:

- stability: preserve the alignment between modalities;
- plasticity: add task-specific discriminative information;
- calibration: avoid overclaiming gains when a backbone is already saturated.

## 12. Strongest Claims to Make

The report can safely make these claims:

1. Real CLAP embeddings exhibit an audio-text modality gap on AudioCaps.
2. MSCLAP has a larger measured gap and stronger cone structure than LAION-CLAP in the current results.
3. Gap manipulation changes contrastive loss, and the loss-preferred gap depends on temperature.
4. Gap shifting improves LAION-CLAP zero-shot ESC-50 accuracy but has almost no effect on the already strong MSCLAP baseline.
5. Naive sequential audio-adapter training is harmful in ESC-50 continual learning.
6. The combined preservation-plus-compensation method is the strongest continual-learning method across both backbones by average and final seen-class accuracy.
7. Phase 2 checks show that the combined method remains strong under fixed beta, multiple seeds, and shuffled class orders.
8. Stricter preservation improves LAION `mgp_mgc` in the marginal alpha summary averaged over tested ranks.
9. The combined method is not highly sensitive to adapter rank in the marginal rank summary averaged over tested alphas.

## 13. Claims to Avoid

Avoid these claims:

1. Do not claim this is full CLAP continual fine-tuning. It is a frozen-embedding adapter study.
2. Do not claim beta was selected using a clean held-out validation protocol unless such a protocol is added.
3. Do not claim preservation alone always beats zero-shot CLAP. It is mixed and backbone-dependent.
4. Do not claim gap shifting universally improves zero-shot accuracy. MSCLAP is essentially flat.
5. Do not claim Figure 2 proves real pretrained CLAP cone behavior. It is a supporting random/synthetic diagnostic.
6. Do not use synthetic folders as main evidence for real-data claims.
7. Do not claim the loss landscape proves actual training convergence. It is a controlled probe.
8. Do not equate strongest accuracy with lowest forgetting; forgetting proxy values should be discussed as a separate metric.
9. Do not describe the LAION alpha and rank summaries as isolated one-variable sweeps; they are marginal summaries from the 4x4 alpha-rank grid.

## 14. Recommended Tables and Figures for the Report

Use these as main report artifacts:

| Report location | Artifact | Source |
|---|---|---|
| Modality-gap section | Gap statistics table | `results/figure_1/*/stats.json` |
| Modality-gap section | PCA/gap visualization | `results/presentation/slide7_modality_gap_pca_*.png` |
| Loss-landscape section | Temperature minima table | `results/presentation/slide8_fig3_minima_table.csv` |
| Loss-landscape section | Simplified Figure 3 plot | `results/presentation/slide8_fig3_simplified_plot.png` |
| Shift section | ESC-50 shift curves | `results/table_1/laion-shift/shift_sweep.png`; `results/table_1/msclap-shift/shift_sweep.png` |
| Continual-learning section | Fixed-beta main table | `results/fixed_beta4_main_table.csv` |
| Robustness section | Seed/order plot | `results/presentation_final/slide14_seed_order_robustness.png` |
| Ablation section | Alpha/rank ablation plot | `results/presentation_final/slide15_alpha_rank_ablation.png` |

Supporting or appendix artifacts:

| Appendix location | Artifact | Source |
|---|---|---|
| Cone diagnostics | Random-init cone plots | `results/figure_2/` |
| Full result provenance | Project status | `results/PROJECT_STATUS.md` |
| Phase 2 provenance | Phase 2 status | `results/PHASE2_STATUS.md` |

## 15. Possible Abstract

> Multimodal contrastive models such as CLAP align audio and text in a shared embedding space, but this shared space can still contain a global modality gap between audio and text representations. We investigate whether the modality-gap analysis and preserve-plus-compensate continual-learning principle from MG-CLIP transfer to audio-text CLAP. Using real AudioCaps embeddings from LAION-CLAP and MSCLAP, we find measurable audio-text modality gaps, with MSCLAP showing a larger centroid separation. Controlled gap-shift probes show that InfoNCE loss depends on the gap and that the loss-preferred gap changes with temperature. On ESC-50, gap shifting improves LAION-CLAP zero-shot accuracy from 0.8310 to 0.8550, while MSCLAP remains nearly saturated. We then introduce MG-CLAP-lite, a frozen-embedding continual-learning method that combines audio-side adapter preservation with audio-prototype compensation. In class-incremental ESC-50, naive adaptation substantially degrades performance, whereas preservation plus compensation achieves the strongest average and final seen-class accuracy across both CLAP backbones and remains robust under fixed beta, multiple seeds, and shuffled class orders. These results suggest that audio-text continual adaptation should preserve cross-modal geometry while adding task-specific audio discrimination, though the conclusions are limited to frozen embeddings rather than full CLAP fine-tuning.

## 16. Possible Conclusion

> The results support a cautious but meaningful transfer of the MG-CLIP story to CLAP. CLAP embeddings show a real audio-text modality gap, and the contrastive objective is sensitive to manipulating that gap. The downstream consequences are backbone-dependent: LAION-CLAP benefits from gap shifting, while MSCLAP is already highly accurate and changes little. The strongest practical finding is in continual learning, where naive audio adaptation harms performance but the combined preservation-plus-compensation strategy is consistently strongest by average and final seen-class accuracy. This suggests that the geometry of pretrained audio-text representations should be treated as useful structure, not incidental noise, when designing continual-learning methods.

## 17. Final Report Thesis

A concise thesis statement for the report is:

> MG-CLAP-lite shows that the modality-gap principle from MG-CLIP transfers to audio-text CLAP at the frozen-embedding adapter level: CLAP has measurable audio-text gap geometry, this geometry interacts with contrastive loss and zero-shot behavior, and preserving it while adding audio-space compensation yields robust average and final accuracy gains on ESC-50.

