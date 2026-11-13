# Codex agent handoff: FAIR1M/FineAir super-resolution experiments

Updated: 2026-10-29

## Purpose and current conclusion

The project investigated whether task-aware super-resolution (SR) could improve fine-grained object classification in low-resolution FAIR1M satellite imagery by using higher-quality imagery from xView and later FineAir as semantic or paired supervision.

The central empirical conclusion is negative but useful: **in-domain training, correct object scale, and tighter crops help substantially, while the learned SR output does not improve classification over the matched LR input or bicubic interpolation.** The final controlled aircraft experiment produced mean test accuracy of 85.44% for learned 2x SR versus 86.86% for direct LR. SR also lost on balanced accuracy and macro F1.

Do not continue open-ended loss tuning without changing the experimental premise. The only remaining experiment considered scientifically defensible without new paired data is a dual-branch LR+SR classifier with learned feature gating, where the original LR evidence is never replaced.

## Infrastructure and privacy

- Workspace: `/workspace`
- Runtime datasets: `/datasets`
- Run outputs: `/runs`
- FineAir host source originally provided by the user: `/mnt/2tb-1/satellitepy/data/FR24_chips/`
- Container FineAir mount used by scripts: `/datasets/FineAir/FR24_chips`
- W&B key: `/workspace/private/wandb_api.txt`
- **Never print, log, commit, or share the W&B key.** Launcher scripts read it into `WANDB_API_KEY` without displaying it.
- Multi-GPU training uses `CUDA_VISIBLE_DEVICES=0,1,2,3` and PyTorch `DataParallel`.
- Data-loader workers above 8 caused several silent first-batch terminations. Four workers was stable for the final SR seeds.

There are no active SR or classifier processes at the time of this handoff.

## Aircraft dataset

Eight overlapping FineAir/FAIR1M aircraft classes were used:

1. A220
2. A321
3. A330
4. A350
5. Boeing737
6. Boeing747
7. Boeing777
8. Boeing787

Prepared dataset root:

`/datasets/fineair_fair1m_aircraft_sr`

Important subdirectories:

- `fineair_hr/{train,validation,test}`: FineAir chips padded to a 320-pixel canvas.
- `fair1m_lr/{train,validation,test}`: initial FAIR1M crops with orientation normalization.
- `fair1m_lr_unoriented/{train,validation,test}`: preferred FAIR1M crops preserving original orientation.

Dataset preparation script:

`/workspace/tools/build_fineair_fair1m_aircraft_sr.py`

Unoriented FAIR1M re-export:

`/workspace/tools/build_fair1m_unoriented_aircraft_chips.py`

The split is disjoint by FAIR1M source image/scene identifier. It is not proven airport/location-disjoint; nearby tiles from the same airport may remain across splits. Treat very high within-FAIR1M accuracy cautiously until stricter geographic grouping is available.

## Object scale finding

Aircraft long-axis lengths were measured from FineAir and FAIR1M oriented bounding boxes. FAIR1M objects require approximately 2.5x apparent enlargement to match FineAir:

- Class-balanced mean ratio: 2.51x
- Class-balanced median ratio: 2.59x
- Per-class mean-ratio range: approximately 2.35x to 2.64x
- Pooled FineAir mean long axis: approximately 166 pixels
- Pooled FAIR1M mean long axis: approximately 67 pixels

The practical scale-normalized classifier therefore used a centered 128x128 FAIR1M crop resized to a 320x320 ResNet input. Orientation was intentionally left unnormalized in the final experiments because earlier automatic nose-direction estimates were unreliable.

## FineAir classifier baseline

Run:

`/runs/fineair_8class_resnet50_x2`

FineAir HR held-out test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 76.62% |
| Balanced accuracy | 66.54% |
| Macro precision | 65.32% |
| Macro F1 | 64.49% |

The relatively low balanced accuracy reflects weak minority-class performance and a small number of independent FineAir scenes. This result is not directly comparable to the within-FAIR1M classifier results because the datasets and split structures differ.

Launcher:

`/workspace/tools/launch_fineair_resnet50_x2.sh`

## Direct FAIR1M LR classifiers

Runs:

- Native 320-pixel chip: `/runs/fair1m_lr_resnet50_direct_native`
- Center 128-pixel crop enlarged to 320: `/runs/fair1m_lr_resnet50_direct_scale25`

Results:

| Input | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Native FAIR1M chip | 85.39% | 85.49% | 85.25% |
| 2.5x center crop | 87.52% | 87.79% | 87.75% |

This demonstrated that the earlier FineAir-to-FAIR1M failure was primarily a domain-transfer problem, not simply insufficient resolution. The 2.5x crop improved the native FAIR1M result by about 2.1 accuracy points by reducing background and matching object scale.

Classifier implementation:

`/workspace/tools/train_xview_resnet50_classifier.py`

It now supports both `--center-crop-size` and `--pre-resize-size`.

## FineAir-guided SR training

### Initial 2x run

Run:

`/runs/fineair_fair1m_sr_x2`

Stopped after epoch 11. Reconstruction and consistency improved, but the SR classifier diagnostic reached only about 17.2%.

### Feature-guided rescaled 2x run

Run:

`/runs/fineair_fair1m_sr_x2_feature_rescaled`

Completed through epoch 40. This branch reduced consistency weight and increased feature/contrastive guidance. Its internal training classifier diagnostic rose to about 25.2%, but that diagnostic was not a held-out classifier result.

Best/final checkpoint used later:

`/runs/fineair_fair1m_sr_x2_feature_rescaled/mambairv2_epoch_0040.pth`

Configuration:

`/workspace/options/train/fineair_fair1m_sr_x2_feature_rescaled.yml`

### Feature-guided 3x run

Run:

`/runs/fineair_fair1m_sr_x3_feature_rescaled`

Paused after the completed epoch-14 checkpoint:

`/runs/fineair_fair1m_sr_x3_feature_rescaled/mambairv2_epoch_0014.pth`

The 3x run was not resumed because the final controlled 2x comparison was prioritized.

## Final controlled aircraft benchmark

Dataset root:

`/datasets/fair1m_final_sr_benchmark`

All variants use the **same centered 128x128 pixels**, the same unoriented FAIR1M splits, identical ResNet-50 training settings, and a 320x320 classifier input:

- `lr128`: direct LR crop, resized by classifier preprocessing
- `bicubic_x2`: the same LR crop bicubically enlarged to 256x256
- `sr_x2`: the same LR crop processed by epoch-40 learned 2x SR

Export script:

`/workspace/tools/export_fair1m_final_sr_benchmark.py`

Run root:

`/runs/fair1m_final_sr_benchmark`

### Per-seed results

| Variant | Seed | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| LR | 10 | 87.52% | 87.79% | 87.75% |
| LR | 20 | 85.22% | 86.25% | 86.18% |
| LR | 30 | 87.85% | 88.22% | 88.16% |
| Bicubic 2x | 10 | 87.85% | 88.57% | 88.69% |
| Bicubic 2x | 20 | 87.68% | 88.59% | 88.73% |
| Learned SR 2x | 10 | 85.88% | 85.34% | 85.36% |
| Learned SR 2x | 20 | 85.88% | 85.24% | 84.94% |
| Learned SR 2x | 30 | 84.56% | 85.24% | 84.82% |

Bicubic seed 30 was intentionally dropped after repeated first-batch worker failures and a user request to prioritize SR results.

### Aggregate interpretation

| Variant | Mean accuracy | Mean balanced accuracy | Mean macro F1 |
|---|---:|---:|---:|
| LR, 3 seeds | 86.86% | 87.42% | 87.37% |
| Bicubic 2x, 2 seeds | 87.77% | 88.58% | 88.71% |
| Learned SR 2x, 3 seeds | 85.44% | 85.27% | 85.04% |

Learned SR loses approximately 1.42 accuracy points and 2.15 balanced-accuracy points relative to direct LR. It also trails both completed bicubic seeds. This is consistent across all three SR seeds and is sufficient evidence to stop claiming that the present learned SR improves aircraft classification.

SR-only launchers retained for reproducibility:

- `/workspace/tools/run_fair1m_final_sr_only.sh`
- `/workspace/tools/run_fair1m_final_sr_remaining_seeds.sh`
- `/workspace/tools/resume_fair1m_final_sr_benchmark.sh`

## Earlier FAIR1M/xView experiments

Before FineAir, the project used xView and a 13-class FAIR1M task. Major stages were:

1. Contrastive unpaired SR with LR consistency.
2. Classifier cross-entropy and total variation.
3. LR-teacher distillation.
4. Geometry preservation via gradient, structure, residual, and equivariance constraints.
5. Strict downsampling consistency.
6. Gradual paired xView mixtures of 10% and 15%.
7. Paired-only sanity training.
8. Analytical and adversarial degradation fitting.

Key conclusion: strict consistency suppressed artifacts but made SR increasingly LR-like. A 10% paired mixture occasionally helped a four-class vehicle subset, but the gain reversed at 15%, so it was not robust.

Important legacy runs include:

- `/runs/fair1m_xview_sr_resnet50`
- `/runs/fair1m_xview_sr_geometry`
- `/runs/fair1m_xview_sr_mixed50`
- `/runs/fair1m_sr_consistency_e40_e43`
- `/runs/fair1m_xview_sr_mix10_fitted_degradation`
- `/runs/fair1m_xview_sr_mix15_fitted_degradation`
- `/runs/paired_xview_sanity`
- `/runs/geometry_constrained_degradation_gan`
- `/runs/geometry_constrained_degradation_gan_v2`
- `/runs/geometry_constrained_degradation_v3`
- `/runs/classifier_feature_matched_degradation_v4`

The paired sanity experiment is important: on known synthetic degradation, paired continuation reached approximately 32.01 dB PSNR and 0.900 SSIM versus bicubic at 25.44 dB and 0.812 SSIM. Thus MambaIRv2 has sufficient capacity when the degradation and HR target are correct. The bottleneck is supervision/domain mismatch, not raw model capacity.

## Why the approach failed

1. **The inverse problem is underdetermined.** Fine subtype details are frequently absent from FAIR1M LR pixels.
2. **FineAir/xView and FAIR1M are different domains.** They differ in sensor response, acquisition, ground sampling, scale, backgrounds, and location distribution.
3. **Unpaired semantic references are not pixel truth.** The model can learn class-plausible texture that is incorrect for the observed aircraft.
4. **Classifier guidance encourages shortcuts.** It can reward feature patterns without improving real geometry.
5. **Strict consistency and perceptual enhancement conflict.** Strong consistency makes the output conservative; weak consistency permits artifacts and hallucinations.
6. **The in-domain LR baseline is already strong.** At roughly 87% accuracy, only a small error budget remains, making speculative pixel synthesis more likely to hurt than help.

## Recommended next step

Do not replace LR images with SR. If one final architectural test is required, use a dual-branch classifier:

- LR branch receives the original FAIR1M crop.
- SR branch receives the frozen SR output.
- Fuse features through a learned gate or attention block.
- Initialize from the strong LR classifier.
- Keep the LR branch available throughout training so the model can ignore harmful SR features.
- Evaluate three seeds on the same fixed test split.
- Continue only if the fused model improves balanced accuracy consistently by at least 1--2 points and the gate assigns non-negligible weight to SR.

The more scientifically reliable alternative is to acquire genuinely paired or well-registered multi-resolution aircraft imagery. Without paired observations or a validated physical sensor model, do not claim that generated detail is a faithful reconstruction.

## Quarterly report and figures

Main LaTeX report:

`/workspace/docs/quarterly_report.tex`

Assets:

`/workspace/docs/quarterly_report_assets`

Text-bearing montage assets were deterministically rebuilt with larger labels. The confusion matrices were changed from side-by-side to vertically stacked for print readability.

Relevant scripts:

- `/workspace/tools/enlarge_quarterly_report_figure_text.py`
- `/workspace/tools/plot_confusion_matrix_comparison.py`

`pdflatex` was not installed in the container, so the updated report was not compiled locally after the figure changes.

## Operational cautions for the next agent

- Inspect existing `/runs` outputs before launching anything; do not overwrite user results.
- Use four data-loader workers for the final aircraft classifiers unless shared-memory limits are verified.
- Do not restart the paused 3x run unless explicitly requested.
- Preserve original FAIR1M orientation unless a reliable annotation-derived orientation method is introduced.
- Do not compare FineAir HR test accuracy directly with within-FAIR1M LR accuracy as if it were a resolution-only comparison.
- Distinguish training-time SR classifier diagnostics from held-out classifier evaluations.
- Never expose `/workspace/private/wandb_api.txt`.
