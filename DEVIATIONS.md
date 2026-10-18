# Deviations from the paper

| ID | Paper detail | Our choice | Reason |
|----|--------------|------------|--------|
| D1 | SynHead++ → Biwi+ DA | **300W** source/target/test split | SynHead++ / Biwi+ not registered |
| D2 | 300W-LP rendered poses | Landmark **solvePnP** Euler + pitch+180 unwrap | 300W-LP not shipped; classic 300W only |
| D3 | 35k / 16k iter schedules, batch 84 | Epoch-based Adam, batch 32, 30 epochs | Small 300W (~600 imgs) |
| D4 | Exact MoCo color/blur code | torchvision/PIL jitter + blur | Functional equivalent |
| D5 | 10-seed mean±std on Biwi+ | Single-seed hold-out MAE | Proxy protocol |

## Taken from paper

- ResNet-18 → 3 Euler angles; relative pose consistency for flip/rotation; λ ramp to 0.2; source-only BN on target; grad clip 5; RCRw frontal down-weighting
