# Speaker Notes: Slides 1-18

## Slide 1: Mind the Gap in CLAP
- Takeaway: The project asks whether the MG-CLIP preserve-and-compensate idea transfers from image-text CLIP to audio-text CLAP.
- What to say in 45-75 seconds: Open by framing the talk as a controlled transfer experiment. We are not claiming full CLAP retraining; we are asking whether the principle survives when the modalities change from images and text to sounds and text.
- Figure explanation: The diagram maps image-text CLIP and MG-CLIP to audio-text CLAP and MG-CLAP-lite.
- Transition: Move from the thesis to how CLAP performs zero-shot classification.
- Likely Q&A: Q: Is this a new CLAP model? A: No. It is a frozen-embedding adapter method built on existing CLAP embeddings.

## Slide 2: CLAP Zero-Shot Classification
- Takeaway: CLAP classifies by comparing an audio embedding to text-label embeddings.
- What to say in 45-75 seconds: Explain that CLAP gives a shared embedding space. Candidate labels are encoded as text prompts, and prediction is the largest audio-text similarity. This matters because preserving geometry is preserving the classifier itself.
- Figure explanation: The figure shows audio and text encoders, similarity scores, and the argmax rule.
- Transition: Next, introduce the original modality-gap premise from CLIP.
- Likely Q&A: Q: Why use text prompts for ESC-50? A: That is the standard zero-shot CLAP interface; labels become text embeddings.

## Slide 3: Original Mind the Gap Premise
- Takeaway: Pretrained contrastive modalities can occupy offset regions, and that offset may be structural.
- What to say in 45-75 seconds: Describe the centroids and the gap vector. The key point is that the gap is not automatically a defect; training may use it as part of the representation geometry. Continual learning can unintentionally destroy it.
- Figure explanation: The schematic shows two modality clouds, centroids, and the gap distance.
- Transition: Next, show how MG-CLIP responds to that risk.
- Likely Q&A: Q: Is this slide measured data? A: No. It is a conceptual schematic; measured CLAP gap stats come later.

## Slide 4: Original MG-CLIP Response
- Takeaway: MG-CLIP balances stability and plasticity through preservation and compensation.
- What to say in 45-75 seconds: Present the two-part logic: preservation avoids destructive cross-modal drift, while compensation adds a task-specific intra-modal classifier. The combination is the principle we test in audio.
- Figure explanation: The diagram branches into Preserve and Compensate and labels them stability and plasticity.
- Transition: Next, map each image-side component to our audio-side version.
- Likely Q&A: Q: Why not only preserve? A: Preservation stabilizes geometry but may not add enough task discrimination.

## Slide 5: Extension Map
- Takeaway: MG-CLAP-lite is a controlled embedding-level extension of MG-CLIP to sound classes.
- What to say in 45-75 seconds: Walk down the table: image-text becomes audio-text, visual classes become sound classes, visual adaptation becomes an audio embedding adapter, and the intra-modal classifier becomes audio prototypes.
- Figure explanation: The table states the mapping and the scope: not full CLAP fine-tuning.
- Transition: Next, show the evidence stack that tests the mapping.
- Likely Q&A: Q: What is controlled here? A: We keep CLAP encoders frozen and operate on cached embeddings.

## Slide 6: Experimental Evidence Map
- Takeaway: The project tests geometry, mechanism, shift behavior, and continual learning with real AudioCaps and ESC-50 outputs.
- What to say in 45-75 seconds: Use this slide as the roadmap. AudioCaps supports the gap and loss-landscape claims; ESC-50 supports zero-shot shift and continual-learning claims.
- Figure explanation: The pipeline and table connect each question to a dataset, model pair, and output type.
- Transition: Next, start with measured CLAP modality gap evidence.
- Likely Q&A: Q: Are synthetic results in the main evidence? A: No. The main evidence uses real AudioCaps and ESC-50 files.

## Slide 7: Modality Gap in CLAP
- Takeaway: CLAP shows an audio-text modality gap in real AudioCaps embeddings.
- What to say in 45-75 seconds: Explain the PCA and schematic. The exact coordinates are only visualization, but the stats come from real cached embeddings. The point is that CLAP shares the same kind of structural issue as CLIP.
- Figure explanation: Slide 7 shows clouds, centroids, and gap direction for audio and text.
- Transition: Next, ask what the contrastive loss itself prefers.
- Likely Q&A: Q: Does PCA prove the gap? A: PCA visualizes it; the numeric centroid gap supports it.

## Slide 8: Contrastive Loss and Temperature
- Takeaway: The loss-preferred gap changes with temperature in real CLAP landscapes.
- What to say in 45-75 seconds: Explain that each temperature has a saved sweep over lambda. The dot is the minimum-loss point. Lower temperature sharpens contrastive pressure and can prefer a non-zero gap.
- Figure explanation: The plot summarizes gap at minimum loss versus temperature for LAION and MSCLAP.
- Transition: Next, test whether shifting the gap affects ESC-50 classification.
- Likely Q&A: Q: Does this prove training convergence? A: No. It is a controlled landscape probe, not full retraining.

## Slide 9: Gap Shifting on ESC-50
- Takeaway: Gap shifts affect LAION more clearly than MSCLAP, so the shift result is mixed rather than universal.
- What to say in 45-75 seconds: Present this as a diagnostic. LAION improves under a shift; MSCLAP is close to null. This motivates a careful claim: geometry matters, but not identically for every backbone.
- Figure explanation: The slide should show the real ESC-50 shift sweeps from Table 1 outputs.
- Transition: Next, introduce the MG-CLAP-lite method that uses preservation and compensation.
- Likely Q&A: Q: Why keep the mixed result? A: It makes the final claims more credible and avoids overgeneralization.

## Slide 10: MG-CLAP-lite Method
- Takeaway: MG-CLAP-lite adds a low-rank audio adapter and combines text and audio-prototype logits.
- What to say in 45-75 seconds: Walk through the pipeline from cached audio embedding to adapted embedding, then into text classifier and prototype classifier. Preservation is early stopping; compensation is the prototype term.
- Figure explanation: The diagram marks frozen encoders, trainable adapter, stored prototypes, and combined logits.
- Transition: Next, explain how preservation is selected.
- Likely Q&A: Q: What is trainable? A: The low-rank residual adapter; CLAP encoders stay frozen.

## Slide 11: Preservation Probe
- Takeaway: The preservation probe selects an epoch before negative-similarity drift exceeds alpha.
- What to say in 45-75 seconds: Explain the drift formula. It monitors how negative-class similarities move relative to epoch zero. The selected epoch is a practical stopping point for maintaining geometry.
- Figure explanation: The plot shows drift over epochs, alpha, and the selected e-star for representative runs.
- Transition: Next, show why prototypes compensate for remaining task discrimination needs.
- Likely Q&A: Q: Is this an aggregate result? A: No. It is a diagnostic for selected run/fold behavior.

## Slide 12: Prototype Compensation
- Takeaway: Audio prototypes add an intra-modal class anchor alongside text embeddings.
- What to say in 45-75 seconds: Explain that text labels remain useful but audio clusters can provide stronger class-specific anchors after adaptation. Beta controls the contribution of the prototype classifier.
- Figure explanation: The schematic shows clusters, prototypes, text embeddings, and similarity arrows.
- Transition: Next, define the continual-learning protocol.
- Likely Q&A: Q: Are prototypes replay memory? A: No. They are compact class anchors, not stored examples.

## Slide 13: Continual-Learning Setup and Main Result
- Takeaway: Evaluation is among all seen classes without task ID, and MGP+MGC is strongest in the fixed-beta table.
- What to say in 45-75 seconds: Describe 10 tasks with 5 classes each. After task k, classification is over all seen classes. Then highlight the fixed beta=4 result, especially MGP+MGC versus naive adapter.
- Figure explanation: The setup diagram and table define the protocol and main result.
- Transition: Next, check whether that result holds across seeds and class orders.
- Likely Q&A: Q: Are future classes included before they appear? A: No. Evaluation only includes seen classes.

## Slide 14: Seed and Class-Order Robustness
- Takeaway: MGP+MGC remains strong across canonical and shuffled class orders.
- What to say in 45-75 seconds: Point out that seed 0 is canonical and seeds 1-3 are shuffled. Both backbones keep MGP+MGC near the top, so the result is not just a single class order artifact.
- Figure explanation: The two panels plot Avg accuracy by seed for each method and backbone.
- Transition: Next, test sensitivity to preservation threshold and adapter rank.
- Likely Q&A: Q: Is beta tuned per seed? A: No. Beta is fixed at 4 in these robustness checks.

## Slide 15: Alpha and Rank Ablation
- Takeaway: Stricter preservation helps, and rank sensitivity is comparatively mild for MGP+MGC.
- What to say in 45-75 seconds: Explain alpha first: smaller alpha means stricter drift tolerance, and the real LAION ablation shows better Avg/Last and lower forgetting. Then explain rank: performance is fairly stable over the tested ranks.
- Figure explanation: The figure plots Avg and Last, with forgetting values shown in side boxes and a compact table.
- Transition: Next, summarize what evidence supports and what remains mixed.
- Likely Q&A: Q: Is this for both backbones? A: This ablation is LAION; main robustness covers both backbones.

## Slide 16: Claims Evidence Matrix
- Takeaway: The strongest claim is preservation plus compensation; several narrower claims are mixed or unsafe.
- What to say in 45-75 seconds: Use the matrix to calibrate the conclusion. The gap exists and the temperature mechanism transfers. Gap shifting is mixed. Naive adaptation is false as a helpful baseline. The combined method is strong on both backbones.
- Figure explanation: Traffic-light colors separate strong, mixed/partial, and false/unsafe claims.
- Transition: Next, make the limitations explicit.
- Likely Q&A: Q: Why include weak claims? A: To prevent the audience from overreading the result.

## Slide 17: Limitations and Threats
- Takeaway: The scope is frozen-embedding MG-CLAP-lite on ESC-50, not full CLAP continual fine-tuning.
- What to say in 45-75 seconds: Name each limitation and immediately pair it with the next step. This turns critique into a concrete roadmap: LoRA, new datasets, validation tuning, replay baseline, and presentation-safety discipline.
- Figure explanation: The table lists limitation, why it matters, and next step.
- Transition: Next, close with the final constrained conclusion.
- Likely Q&A: Q: Do these limitations invalidate the result? A: No. They define the scope of the controlled transfer result.

## Slide 18: Final Takeaway
- Takeaway: Geometry, mechanism, and the preserve-plus-compensate continual-learning principle transfer at the embedding-adapter level.
- What to say in 45-75 seconds: Close with three blocks: CLAP has the gap, temperature changes the loss-preferred gap, and preservation plus compensation beats naive adaptation. Then restate the warning: not full CLAP fine-tuning.
- Figure explanation: The slide has three conclusion blocks and a scope warning.
- Transition: End by inviting questions around scope, baselines, and next experiments.
- Likely Q&A: Q: What is the single safest final claim? A: MG-CLAP-lite supports the MG-CLIP principle at the frozen-embedding adapter level.
