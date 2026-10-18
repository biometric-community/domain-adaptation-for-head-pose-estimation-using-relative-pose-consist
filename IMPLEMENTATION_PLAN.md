# Implementation plan: HPE relative pose consistency

Paper: Kuhnke & Ostermann, TBIOM 2023, DOI 10.1109/TBIOM.2023.3237039.

## Scope

ResNet-18 + MSE + RCRw on **300W** (solvePnP poses). Biwi+/SynHead++/300W-LP blocked (D1–D2).

## Size gate

| Dataset | Bytes | GiB | Decision |
|---------|------:|----:|----------|
| `projects/datasets/300w` | 4285214098 | 3.9909 | **full_train** |

## Figure inventory (paper Results → ours)

| Paper fig | Type | Our artifact |
|-----------|------|--------------|
| Angle MAE bars / tables | bar | `outputs/figures/mae_angles.svg` |
| Training / λ schedules (spirit) | line | `outputs/figures/train_curves.svg` |

Skip method cartoons (Figs. 1–3 architecture).
