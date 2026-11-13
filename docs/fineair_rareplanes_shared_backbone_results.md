# FineAir–RarePlanes shared-backbone results

Seed 0, YOLO11s OBB, 1024 px, global batch 8. Test metrics use each dataset's
untouched test split and full single-GPU evaluation.

## Detection mAP50–95

| Variant | FineAir independent | FineAir shared | Delta | RarePlanes independent | RarePlanes shared | Delta | Mean independent | Mean shared | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint semantics | 0.9145 | 0.9190 | +0.0045 | 0.7818 | 0.7760 | -0.0058 | 0.8481 | 0.8475 | -0.0007 |
| Detached semantics | 0.9372 | 0.9350 | -0.0022 | 0.7865 | 0.8025 | +0.0160 | 0.8619 | **0.8687** | **+0.0069** |
| Detection only | 0.9385 | 0.9221 | -0.0164 | 0.7826 | 0.7934 | +0.0108 | 0.8605 | 0.8577 | -0.0028 |

## Interpretation

- Detached semantic training is the strongest shared-backbone baseline. It
  transfers useful detector features to RarePlanes while nearly preserving
  FineAir detection.
- Fully joint semantic gradients erase the mean detection benefit. This is
  evidence of semantic gradient interference, not a general failure of the
  shared detector.
- Detection-only sharing favors RarePlanes but costs substantially more on
  FineAir, suggesting the sparse semantic tower provides useful regularization
  even when its gradients are detached.
- Results are from one seed; deltas should be replicated before treating small
  differences as statistically stable.

## Next experiment

The next controlled run starts from the detached mechanism and adds a small
task-specific residual adapter at each P3/P4/P5 semantic input. A learned gate
controls the only semantic gradient path into shared features, with a penalty
that favors closed gates. This tests whether individual tasks can recover the
classification gains of joint training without its detection penalty.
