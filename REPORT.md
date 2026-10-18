# Report: Domain Adaptation for Head Pose Estimation (RCRw)

Rebuild of Kuhnke & Ostermann, TBIOM 2023 (relative pose consistency). DOI from paper folder analysis.

## Setup

- ResNet-18 → 3 Euler angles; Adam lr=0.0001; λ→0.2
- Data: 300W (landmark→solvePnP poses); size gate full_train (3.9909 GiB)
- Biwi+ / SynHead++ / 300W-LP not registered — see `DEVIATIONS.md`

## Hold-out results (degrees)

| | MAE | pitch | yaw | roll |
|--|--:|--:|--:|--:|
| **Ours** | 22.81 | 16.31 | 21.98 | 30.13 |

## Figures

### `outputs/figures/mae_angles.svg`

![](outputs/figures/mae_angles.svg)

### `outputs/figures/train_curves.svg`

![](outputs/figures/train_curves.svg)

## Regenerate

```bash
bash scripts/report.sh
```
