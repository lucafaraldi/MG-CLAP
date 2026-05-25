# Slide 11 explanation

The preservation probe uses saved `preservation_probe.csv` files. Phase 2 seed-0 probes are preferred when available; otherwise the base continual MG-CLAP probes are used.

The plotted drift is the saved relative negative-similarity drift:

`D_e = |neg_e - neg_0| / max(|neg_0|, epsilon)`

The dashed horizontal line is `alpha` from the corresponding `summary.json`. The vertical line is the selected epoch `e_star`, read from the CSV `selected` flag when present.

Interpretation:

- Preservation is implemented as early stopping against negative-class similarity drift.
- The adapter can improve positive alignment while stopping before negative similarities move too far from the frozen CLAP geometry.
- This plot is a diagnostic for the selected run/fold, not an aggregate performance result.
