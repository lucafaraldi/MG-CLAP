# Slide 14 speaker notes

- Fixed beta=4 robustness is evaluated across four seeds per backbone.
- Seed 0 uses the canonical ESC-50 class order; seeds 1-3 use shuffled class orders.
- MGP+MGC is consistently near the top for both LAION and MSCLAP.
- Caveat: shuffled class orders are robustness probes, not a replacement for new datasets.
- Caveat: beta is fixed here, not tuned on a held-out validation split.
