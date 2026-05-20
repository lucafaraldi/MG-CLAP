# MG-CLAP Project Status

- Timestamp: `2026-05-20T21:49:43.418265+00:00`
- Git commit: `8f3ff755e36f4c5eff20d8648a34c25e461a3c52`

## Result Classification

| result | status | safe? | notes |
|---|---|---|---|
| figure_1/laion-audiocaps-val | real-data | yes | Real AudioCaps gap stats. |
| figure_1/msclap-audiocaps-val | real-data | yes | Real AudioCaps gap stats. |
| figure_3/laion-audiocaps-val | real-data | yes | Real-cache Figure 3 run. |
| figure_3/msclap-audiocaps-val | real-data | yes | Real-cache Figure 3 run. |
| figure_3/synth-gap0.82 | synthetic | no | Synthetic Figure 3 run. |
| table_1/simulation | synthetic | no | Synthetic zero-shot shift sweep. |
| table_1/laion-shift | real-data | yes | Real ESC-50 shift run. |
| table_1/msclap-shift | real-data | yes | Real ESC-50 shift run. |
| table_1_training/laion | unclear-provenance | no | Training sweep exists but provenance may be mixed or absent. |
| continual_mgclap/laion | real-data | yes | MG-CLAP-lite continual-learning run. |
| continual_mgclap/laion/dry_run | real-data | no | Dry run only. |
| continual_mgclap/msclap | real-data | yes | MG-CLAP-lite continual-learning run. |

## Presentation Safety

### Safe for Presentation

- figure_1/laion-audiocaps-val
- figure_1/msclap-audiocaps-val
- figure_3/laion-audiocaps-val
- figure_3/msclap-audiocaps-val
- table_1/laion-shift
- table_1/msclap-shift
- continual_mgclap/laion
- continual_mgclap/msclap

### Not Safe for Presentation

- figure_3/synth-gap0.82
- table_1/simulation
- table_1_training/laion
- continual_mgclap/laion/dry_run

## Explicit Answers

- Real ESC-50 continual learning exists: `True`
- Real ESC-50 shift exists: `True`
- Real CLAP Figure 3 exists: `True`

## Remaining Work

- figure_3/synth-gap0.82
- table_1/simulation
- table_1_training/laion
