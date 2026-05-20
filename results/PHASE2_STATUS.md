# MG-CLAP Phase 2 Status

- Timestamp: `2026-05-20T23:02:54.196913+00:00`
- Git commit: `8f3ff755e36f4c5eff20d8648a34c25e461a3c52`
- Main-grid runs found: `8`
- Alpha/rank runs found: `16`

## What Was Run

- Phase 2 main grid: 8/8 runs found.
- LAION alpha/rank ablation: 16/16 runs found.

## What Remains To Run

- Nothing missing from the requested Phase 2 set.

## Fixed-Beta=4 Main Table

| backbone | method | runs | Avg | Last | forgetting |
|---|---|---:|---:|---:|---:|
| laion | mgc_only | 4 | 0.8104 ± 0.0194 | 0.7095 ± 0.0573 | 0.3157 ± 0.0639 |
| laion | mgp_mgc | 4 | 0.9694 ± 0.0156 | 0.9417 ± 0.0285 | 0.0511 ± 0.0309 |
| laion | mgp_only | 4 | 0.8994 ± 0.0423 | 0.8683 ± 0.0370 | 0.0993 ± 0.0511 |
| laion | naive_adapter | 4 | 0.7721 ± 0.0307 | 0.6629 ± 0.0604 | 0.3685 ± 0.0655 |
| laion | zero_shot_clap | 4 | 0.8685 ± 0.0303 | 0.8310 ± 0.0000 | 0.0565 ± 0.0166 |
| msclap | mgc_only | 4 | 0.8388 ± 0.0394 | 0.7924 ± 0.0333 | 0.2251 ± 0.0397 |
| msclap | mgp_mgc | 4 | 0.9731 ± 0.0089 | 0.9570 ± 0.0101 | 0.0404 ± 0.0126 |
| msclap | mgp_only | 4 | 0.9545 ± 0.0209 | 0.9365 ± 0.0151 | 0.0597 ± 0.0169 |
| msclap | naive_adapter | 4 | 0.8188 ± 0.0442 | 0.7720 ± 0.0197 | 0.2482 ± 0.0251 |
| msclap | zero_shot_clap | 4 | 0.9685 ± 0.0082 | 0.9435 ± 0.0000 | 0.0274 ± 0.0059 |

## By Seed

| backbone | seed | order | method | Avg | Last | forgetting |
|---|---:|---|---|---:|---:|---:|
| laion | 0 | canonical | mgc_only | 0.7797 | 0.7235 | 0.2944 |
| laion | 0 | canonical | mgp_mgc | 0.9453 | 0.9040 | 0.0917 |
| laion | 0 | canonical | mgp_only | 0.8273 | 0.8060 | 0.1856 |
| laion | 0 | canonical | naive_adapter | 0.7240 | 0.6255 | 0.4039 |
| laion | 0 | canonical | zero_shot_clap | 0.8176 | 0.8310 | 0.0444 |
| laion | 1 | shuffled | mgc_only | 0.8078 | 0.7965 | 0.2217 |
| laion | 1 | shuffled | mgp_mgc | 0.9780 | 0.9680 | 0.0222 |
| laion | 1 | shuffled | mgp_only | 0.9124 | 0.9015 | 0.0539 |
| laion | 1 | shuffled | naive_adapter | 0.7673 | 0.7650 | 0.2578 |
| laion | 1 | shuffled | zero_shot_clap | 0.8737 | 0.8310 | 0.0756 |
| laion | 2 | shuffled | mgc_only | 0.8266 | 0.6475 | 0.3878 |
| laion | 2 | shuffled | mgp_mgc | 0.9869 | 0.9705 | 0.0200 |
| laion | 2 | shuffled | mgp_only | 0.9240 | 0.8770 | 0.0711 |
| laion | 2 | shuffled | naive_adapter | 0.8036 | 0.6125 | 0.4261 |
| laion | 2 | shuffled | zero_shot_clap | 0.8890 | 0.8310 | 0.0700 |
| laion | 3 | shuffled | mgc_only | 0.8274 | 0.6705 | 0.3589 |
| laion | 3 | shuffled | mgp_mgc | 0.9674 | 0.9245 | 0.0706 |
| laion | 3 | shuffled | mgp_only | 0.9339 | 0.8885 | 0.0867 |
| laion | 3 | shuffled | naive_adapter | 0.7933 | 0.6485 | 0.3861 |
| laion | 3 | shuffled | zero_shot_clap | 0.8938 | 0.8310 | 0.0361 |
| msclap | 0 | canonical | mgc_only | 0.7749 | 0.8220 | 0.1872 |
| msclap | 0 | canonical | mgp_mgc | 0.9657 | 0.9485 | 0.0483 |
| msclap | 0 | canonical | mgp_only | 0.9240 | 0.9135 | 0.0839 |
| msclap | 0 | canonical | naive_adapter | 0.7474 | 0.7955 | 0.2178 |
| msclap | 0 | canonical | zero_shot_clap | 0.9558 | 0.9435 | 0.0178 |
| msclap | 1 | shuffled | mgc_only | 0.8382 | 0.7395 | 0.2872 |
| msclap | 1 | shuffled | mgp_mgc | 0.9629 | 0.9550 | 0.0444 |
| msclap | 1 | shuffled | mgp_only | 0.9461 | 0.9375 | 0.0606 |
| msclap | 1 | shuffled | naive_adapter | 0.8299 | 0.7435 | 0.2839 |
| msclap | 1 | shuffled | zero_shot_clap | 0.9688 | 0.9435 | 0.0300 |
| msclap | 2 | shuffled | mgc_only | 0.8762 | 0.7885 | 0.2322 |
| msclap | 2 | shuffled | mgp_mgc | 0.9811 | 0.9505 | 0.0500 |
| msclap | 2 | shuffled | mgp_only | 0.9739 | 0.9390 | 0.0583 |
| msclap | 2 | shuffled | naive_adapter | 0.8686 | 0.7650 | 0.2578 |
| msclap | 2 | shuffled | zero_shot_clap | 0.9786 | 0.9435 | 0.0339 |
| msclap | 3 | shuffled | mgc_only | 0.8658 | 0.8195 | 0.1939 |
| msclap | 3 | shuffled | mgp_mgc | 0.9828 | 0.9740 | 0.0189 |
| msclap | 3 | shuffled | mgp_only | 0.9739 | 0.9560 | 0.0361 |
| msclap | 3 | shuffled | naive_adapter | 0.8295 | 0.7840 | 0.2333 |
| msclap | 3 | shuffled | zero_shot_clap | 0.9709 | 0.9435 | 0.0278 |

## Class-Order Robustness

| backbone | method | canonical Avg | shuffled Avg | delta Avg | canonical Last | shuffled Last | delta Last |
|---|---|---:|---:|---:|---:|---:|---:|
| laion | mgc_only | 0.7797 | 0.8206 ± 0.0091 | 0.0410 | 0.7235 | 0.7048 ± 0.0655 | -0.0187 |
| laion | mgp_mgc | 0.9453 | 0.9774 ± 0.0080 | 0.0322 | 0.9040 | 0.9543 ± 0.0211 | 0.0503 |
| laion | mgp_only | 0.8273 | 0.9234 ± 0.0088 | 0.0961 | 0.8060 | 0.8890 ± 0.0100 | 0.0830 |
| laion | naive_adapter | 0.7240 | 0.7881 ± 0.0153 | 0.0641 | 0.6255 | 0.6753 ± 0.0651 | 0.0498 |
| laion | zero_shot_clap | 0.8176 | 0.8855 ± 0.0086 | 0.0679 | 0.8310 | 0.8310 ± 0.0000 | 0.0000 |
| msclap | mgc_only | 0.7749 | 0.8601 ± 0.0160 | 0.0852 | 0.8220 | 0.7825 ± 0.0329 | -0.0395 |
| msclap | mgp_mgc | 0.9657 | 0.9756 ± 0.0090 | 0.0099 | 0.9485 | 0.9598 ± 0.0102 | 0.0113 |
| msclap | mgp_only | 0.9240 | 0.9646 ± 0.0131 | 0.0406 | 0.9135 | 0.9442 ± 0.0084 | 0.0307 |
| msclap | naive_adapter | 0.7474 | 0.8427 ± 0.0183 | 0.0953 | 0.7955 | 0.7642 ± 0.0165 | -0.0313 |
| msclap | zero_shot_clap | 0.9558 | 0.9728 ± 0.0042 | 0.0169 | 0.9435 | 0.9435 ± 0.0000 | -0.0000 |

## Alpha Sensitivity

| method | alpha | runs | Avg | Last | forgetting |
|---|---:|---:|---:|---:|---:|
| mgc_only | 0.05 | 4 | 0.7851 ± 0.0398 | 0.7294 ± 0.0335 | 0.2878 ± 0.0382 |
| mgc_only | 0.10 | 4 | 0.7881 ± 0.0420 | 0.7324 ± 0.0414 | 0.2844 ± 0.0463 |
| mgc_only | 0.20 | 4 | 0.7862 ± 0.0422 | 0.7295 ± 0.0423 | 0.2869 ± 0.0478 |
| mgc_only | 0.30 | 4 | 0.7860 ± 0.0392 | 0.7237 ± 0.0313 | 0.2935 ± 0.0355 |
| mgp_mgc | 0.05 | 4 | 0.9589 ± 0.0032 | 0.9226 ± 0.0055 | 0.0711 ± 0.0053 |
| mgp_mgc | 0.10 | 4 | 0.9482 ± 0.0032 | 0.9054 ± 0.0041 | 0.0899 ± 0.0046 |
| mgp_mgc | 0.20 | 4 | 0.9131 ± 0.0061 | 0.8575 ± 0.0060 | 0.1431 ± 0.0072 |
| mgp_mgc | 0.30 | 4 | 0.8813 ± 0.0105 | 0.8256 ± 0.0085 | 0.1783 ± 0.0093 |
| mgp_only | 0.05 | 4 | 0.8322 ± 0.0027 | 0.8146 ± 0.0052 | 0.1739 ± 0.0064 |
| mgp_only | 0.10 | 4 | 0.8277 ± 0.0008 | 0.8095 ± 0.0028 | 0.1811 ± 0.0037 |
| mgp_only | 0.20 | 4 | 0.8164 ± 0.0040 | 0.7865 ± 0.0080 | 0.2100 ± 0.0084 |
| mgp_only | 0.30 | 4 | 0.8009 ± 0.0051 | 0.7433 ± 0.0066 | 0.2597 ± 0.0080 |
| naive_adapter | 0.05 | 4 | 0.7391 ± 0.0336 | 0.6415 ± 0.0374 | 0.3835 ± 0.0453 |
| naive_adapter | 0.10 | 4 | 0.7371 ± 0.0350 | 0.6436 ± 0.0406 | 0.3815 ± 0.0488 |
| naive_adapter | 0.20 | 4 | 0.7383 ± 0.0339 | 0.6491 ± 0.0370 | 0.3754 ± 0.0444 |
| naive_adapter | 0.30 | 4 | 0.7420 ± 0.0331 | 0.6531 ± 0.0340 | 0.3714 ± 0.0409 |
| zero_shot_clap | 0.05 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |
| zero_shot_clap | 0.10 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |
| zero_shot_clap | 0.20 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |
| zero_shot_clap | 0.30 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |

## Rank Sensitivity

| method | rank | runs | Avg | Last | forgetting |
|---|---:|---:|---:|---:|---:|
| mgc_only | 4 | 4 | 0.8399 ± 0.0012 | 0.7709 ± 0.0067 | 0.2404 ± 0.0078 |
| mgc_only | 8 | 4 | 0.8022 ± 0.0024 | 0.7481 ± 0.0069 | 0.2661 ± 0.0063 |
| mgc_only | 16 | 4 | 0.7752 ± 0.0032 | 0.7231 ± 0.0123 | 0.2949 ± 0.0141 |
| mgc_only | 32 | 4 | 0.7281 ± 0.0032 | 0.6729 ± 0.0097 | 0.3513 ± 0.0105 |
| mgp_mgc | 4 | 4 | 0.9282 ± 0.0269 | 0.8781 ± 0.0366 | 0.1197 ± 0.0404 |
| mgp_mgc | 8 | 4 | 0.9300 ± 0.0280 | 0.8834 ± 0.0351 | 0.1149 ± 0.0388 |
| mgp_mgc | 16 | 4 | 0.9223 ± 0.0316 | 0.8745 ± 0.0420 | 0.1247 ± 0.0463 |
| mgp_mgc | 32 | 4 | 0.9210 ± 0.0366 | 0.8751 ± 0.0409 | 0.1231 ± 0.0457 |
| mgp_only | 4 | 4 | 0.8224 ± 0.0102 | 0.7911 ± 0.0292 | 0.2037 ± 0.0352 |
| mgp_only | 8 | 4 | 0.8208 ± 0.0119 | 0.7856 ± 0.0342 | 0.2090 ± 0.0408 |
| mgp_only | 16 | 4 | 0.8181 ± 0.0122 | 0.7865 ± 0.0256 | 0.2082 ± 0.0305 |
| mgp_only | 32 | 4 | 0.8158 ± 0.0146 | 0.7906 ± 0.0249 | 0.2037 ± 0.0297 |
| naive_adapter | 4 | 4 | 0.7799 ± 0.0015 | 0.6938 ± 0.0037 | 0.3204 ± 0.0040 |
| naive_adapter | 8 | 4 | 0.7599 ± 0.0020 | 0.6669 ± 0.0023 | 0.3549 ± 0.0028 |
| naive_adapter | 16 | 4 | 0.7258 ± 0.0012 | 0.6320 ± 0.0074 | 0.3967 ± 0.0082 |
| naive_adapter | 32 | 4 | 0.6909 ± 0.0033 | 0.5947 ± 0.0066 | 0.4399 ± 0.0073 |
| zero_shot_clap | 4 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |
| zero_shot_clap | 8 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |
| zero_shot_clap | 16 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |
| zero_shot_clap | 32 | 4 | 0.8176 ± 0.0000 | 0.8310 ± 0.0000 | 0.0444 ± 0.0000 |

## Which Claims Become Stronger

- Backbone-level conclusions are no longer tied to a single seed/order run.
- The fixed-β=4 combined preservation+compensation result can be assessed for robustness across seeds and shuffled class orders.
- LAION sensitivity to alpha and adapter rank can be discussed with real ablation data rather than a single default configuration.

## Which Claims Remain Unsafe

- Do not claim full CLAP continual fine-tuning; Phase 2 still evaluates a frozen-embedding adapter setup.
- Do not claim beta is validation-tuned; Phase 2 fixes beta=4 for robustness checks but does not introduce a held-out tuning split.
- Do not claim Figure 2 is real-data validated.
