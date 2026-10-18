# Fidelity audit: HPE RCRw

Paper: Kuhnke & Ostermann, TBIOM 2023, DOI 10.1109/TBIOM.2023.3237039.

## Pass scorecard

| Pass | Focus | Confidence | Notes |
|------|-------|------------|-------|
| 1 | Layout / scripts / configs | 95% | `hpe_rcr/`, `scripts/{setup,train,train_full,eval,predict,report}.sh`, YAML |
| 2 | RCR + source BN + RCRw | 90% | Relative flip/rot mapping; `set_bn_eval` on target; frontal down-weight; λ→0.2; clip 5 |
| 3 | Protocol / data | 55% | **D1–D2**: 300W solvePnP proxy (not SynHead++→Biwi+ / 300W-LP) |
| 4 | Eval / REPORT / figures | 90% | MAE angles bar + train curves; real metrics only |
| 5 | Smoke + size-gated full | 85% | Smoke OK; 300W 3.99 GiB → full 25 epochs; best MAE≈22.8° |

**Mean confidence (equal weight):** ≈ **83%**. Stop after Pass 5.

## Pass 1 — structure

- Package `hpe_rcr/` with data, models, losses, train/eval/predict/report.
- Shared monorepo `.venv` via `scripts/setup_env.sh`.
- Size gate script + `outputs/logs/dataset_size.json` (`decision=full_train`).

## Pass 2 — algorithm

Taken from paper: ResNet-18→3 Euler; MSE supervised; relative consistency on target with flip (yaw/roll sign) + rotation (roll Δ); RCRw weighting; source-only BN on target; grad clip 5; λ ramp to 0.2.

Gaps: batch 32 / 25 epochs vs paper batch 84 / multi-10k iters (D3); MoCo augs approximated (D4).

## Pass 3 — data protocol

Blocked corpora: SynHead++, Biwi+, 300W-LP. Proxy: classic 300W Indoor/Outdoor (600 images), landmark→solvePnP labels with pitch+180 unwrap, 60/20/20 source/target/test (D1–D2, D5).

## Pass 4 — reporting

`REPORT.md` + `outputs/figures/{mae_angles,train_curves}.svg` from `train_summary` / `eval_results`. No fabricated Biwi+ table numbers.

## Pass 5 — runs

| Run | Result |
|-----|--------|
| Smoke (`configs/smoke.yaml`) | Completes 2 epochs on 80-image subset |
| Full (`configs/default.yaml`) | 360/120/120 split; best MAE **22.80°** (p/y/r ≈ 16.3 / 22.0 / 30.1) @ epoch 3 |

Proxy MAE is **not** comparable to paper Biwi+ RCRw 3.85° (different data + labels).

## Sign-off

Fidelity audit complete (5 passes). Remaining gap is dataset protocol (D1–D2), not missing RCR mechanics.
