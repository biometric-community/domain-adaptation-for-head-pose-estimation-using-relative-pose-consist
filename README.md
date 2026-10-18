# Domain Adaptation for Head Pose Estimation Using Relative Pose Consistency

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of Kuhnke & Ostermann (TBIOM 2023): ResNet-18 head-pose regression with **weighted relative pose consistency (RCRw)** and source-only BatchNorm. Trained on **300W** with landmark-derived Euler labels (Biwi+ / SynHead++ / 300W-LP not shipped — see `DEVIATIONS.md`).

**Repository:** https://github.com/biometric-community/domain-adaptation-for-head-pose-estimation-using-relative-pose-consist

**Paper:** [10.1109/TBIOM.2023.3237039](https://doi.org/10.1109/TBIOM.2023.3237039)

## Quick start

```bash
bash scripts/setup_env.sh
bash scripts/train.sh configs/smoke.yaml          # sanity
bash scripts/train_full.sh                        # size gate → full 300W
bash scripts/eval.sh
bash scripts/report.sh                            # REPORT.md + SVG figures
bash scripts/predict.sh /path/to/face/images
```

Data root (config): `../../datasets/300w` (Indoor/Outdoor `.png` + `.pts`).

## Results (this rebuild)

Hold-out MAE on 300W proxy split (degrees): **22.80** (pitch 16.3 / yaw 22.0 / roll 30.1). Not comparable to paper Biwi+ RCRw ≈3.85°. Details in [`REPORT.md`](REPORT.md).

## Layout

```
hpe_rcr/          # package (data, models, losses, train/eval/report)
configs/          # smoke.yaml, default.yaml
scripts/          # shell entrypoints
outputs/          # checkpoints, logs, figures (gitignored)
```

## Citation

```bibtex
@article{kuhnke2023hpe,
  title={Domain Adaptation for Head Pose Estimation Using Relative Pose Consistency},
  author={Kuhnke, Felix and Ostermann, J{\"o}rn},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  volume={5},
  number={3},
  pages={348--359},
  year={2023},
  doi={10.1109/TBIOM.2023.3237039}
}
```
