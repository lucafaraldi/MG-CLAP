# Slide 8 explanation

Inputs are the real Figure 3b landscape JSON files for LAION-CLAP and MSCLAP on AudioCaps validation. For each backbone and temperature $\tau$, the table selects the saved $\lambda$ value with the lowest InfoNCE loss.

Why original Figure 3b can show two branches:

- The original plot uses gap distance on the x-axis, but the sweep parameter is $\lambda$ along the modality-gap direction.
- Gap distance folds the $\lambda$ sweep: moving toward the centroids and moving past the zero-gap point can yield similar distances.
- Those two sides need not have identical InfoNCE loss because the paired similarities and neighborhood geometry are not symmetric after the shift.

What the coloured dots mean:

- Each coloured curve corresponds to one temperature $\tau$.
- The dot marks the saved sweep point with minimum InfoNCE loss for that temperature.

Safe claim:

- In these real AudioCaps CLAP landscapes, the loss-preferred gap depends on temperature. Lower temperatures can prefer a non-zero gap, while higher temperatures can move the minimum toward smaller gaps.

Claim to avoid:

- Do not claim this sweep proves the trained encoder will converge to exactly these gaps, or that downstream continual-learning performance is caused only by the modality-gap value. The sweep is a controlled loss-landscape probe, not a full training trajectory or causal ablation by itself.
