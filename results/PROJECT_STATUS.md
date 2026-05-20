# MG-CLAP Project Status

- Timestamp: `2026-05-20T19:51:03.078819+00:00`
- Git commit: `8f3ff755e36f4c5eff20d8648a34c25e461a3c52`

## Result Classification

| result | status | safe? | notes |
|---|---|---|---|
| figure_1/laion-audiocaps-val | real-data | yes | Real AudioCaps gap stats. |
| figure_1/msclap-audiocaps-val | real-data | yes | Real AudioCaps gap stats. |
| figure_3/synth-gap0.82 | synthetic | no | Synthetic Figure 3 run. |
| table_1/simulation | synthetic | no | Synthetic zero-shot shift sweep. |
| table_1/laion-shift | missing | no | Missing real shift summary. |
| table_1/msclap-shift | missing | no | Missing real shift summary. |
| table_1_training/laion | unclear-provenance | no | Training sweep exists but provenance may be mixed or absent. |
| continual_mgclap | missing | no | No continual results yet. |

## Presentation Safety

### Safe for Presentation

- figure_1/laion-audiocaps-val
- figure_1/msclap-audiocaps-val

### Not Safe for Presentation

- figure_3/synth-gap0.82
- table_1/simulation
- table_1/laion-shift
- table_1/msclap-shift
- table_1_training/laion
- continual_mgclap

## Explicit Answers

- Real ESC-50 continual learning exists: `False`
- Real ESC-50 shift exists: `False`
- Real CLAP Figure 3 exists: `False`

## Remaining Work

- figure_3/synth-gap0.82
- table_1/simulation
- table_1/laion-shift
- table_1/msclap-shift
- table_1_training/laion
- continual_mgclap
