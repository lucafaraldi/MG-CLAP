# Asset Manifest

Generated final presentation assets.

- Slide 1: `results/presentation_final/slide1_title_thesis.png` - Title visual
- Slide 2: `results/presentation_final/slide2_clap_zero_shot_classifier.png` - CLAP zero-shot classifier schematic
- Slide 3: `results/presentation_final/slide3_original_mind_the_gap_premise.png` - Original Mind the Gap premise
- Slide 4: `results/presentation_final/slide4_original_mgclip_preserve_compensate.png` - Original MG-CLIP preserve/compensate diagram
- Slide 5: `results/presentation_final/slide5_extension_from_mgclip_to_mgclap.png` - Extension map table
- Slide 6: `results/presentation_final/slide6_experimental_evidence_map.png` - Experimental evidence map
- Slide 14: `results/presentation_final/slide14_seed_order_robustness.png` - Seed and class-order robustness plot
- Slide 14: `results/presentation_final/slide14_seed_order_summary.md` - Seed/order speaker-note summary
- Slide 15: `results/presentation_final/slide15_alpha_rank_ablation.png` - Alpha/rank ablation plot
- Slide 15: `results/presentation_final/slide15_alpha_rank_table.png` - Compact alpha/rank numeric table
- Slide 16: `results/presentation_final/slide16_claims_evidence_matrix.png` - Claims/evidence traffic-light matrix
- Slide 17: `results/presentation_final/slide17_limitations_threats.png` - Limitations and threats table
- Slide 18: `results/presentation_final/slide18_final_takeaway.png` - Final conclusion slide
- Backup 1: `results/presentation_final/backup_formula_sheet.png` - Formula sheet
- Backup 2: `results/presentation_final/backup_full_phase2_table.png` - Full fixed beta=4 table
- Backup 3: `results/presentation_final/backup_class_order_table.png` - Class-order robustness table
- Backup 4: `results/presentation_final/backup_rank_sensitivity_table.png` - Rank sensitivity table
- Backup 5: `results/presentation_final/backup_real_vs_synthetic_status.png` - Real vs synthetic presentation-safety table
- Speaker notes: `results/presentation_final/SPEAKER_NOTES_SLIDES_1_18.md` - Slides 1-18 speaker notes
- Manifest: `results/presentation_final/ASSET_MANIFEST.md` - Generated asset manifest

## Numeric Source Policy

Main-slide numeric plots/tables are generated only from these real-data files:
- `results/fixed_beta4_main_table.csv`
- `results/phase2_main_grid_table.csv`
- `results/phase2_alpha_sensitivity.csv`
- `results/phase2_rank_sensitivity.csv`
- `results/phase2_class_order_robustness.csv`
- `results/figure_1/laion-audiocaps-val/stats.json`
- `results/figure_1/msclap-audiocaps-val/stats.json`
- `results/table_1/laion-shift/summary.json`
- `results/table_1/msclap-shift/summary.json`
- `results/figure_3/laion-audiocaps-val/3b_landscape.json`
- `results/figure_3/msclap-audiocaps-val/3b_landscape.json`

Synthetic/historical artifacts excluded from main-slide numeric plots:
- `results/figure_3/synth-gap0.82/`
- `results/table_1/simulation/`
- `results/table_1_training/laion/`
- `results/continual_mgclap/laion/dry_run/`
- `handoff/` archive copies