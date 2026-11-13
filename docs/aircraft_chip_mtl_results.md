# Aircraft chip MTL results

Held-out scene-level test results. Classification columns are unweighted means across the available tasks within each dataset. DOTA Dice is the pixel-level F1 score.

| Run | Seed | Mask route | Weight | FineAir acc. | FineAir macro R | FineAir macro F1 | RarePlanes acc. | RarePlanes macro R | RarePlanes macro F1 | DOTA Dice/F1 | DOTA IoU |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dota_mask_independent | 0 | full | 1.00 | — | — | — | — | — | — | 89.04 | 80.25 |
| dota_mask_independent_early_seed0 | 0 | early | 1.00 | — | — | — | — | — | — | 89.83 | 81.54 |
| dota_mask_independent_early_seed1 | 1 | early | 1.00 | — | — | — | — | — | — | 89.74 | 81.39 |
| dota_mask_independent_early_seed2 | 2 | early | 1.00 | — | — | — | — | — | — | 89.80 | 81.49 |
| dota_mask_independent_seed1 | 1 | full | 1.00 | — | — | — | — | — | — | 88.91 | 80.04 |
| dota_mask_independent_seed2 | 2 | full | 1.00 | — | — | — | — | — | — | 89.04 | 80.25 |
| fineair_independent | 0 | none | 1.00 | 47.92 | 36.47 | 35.17 | — | — | — | — | — |
| fineair_independent_seed1 | 1 | none | 1.00 | 45.85 | 31.60 | 28.05 | — | — | — | — | — |
| fineair_independent_seed2 | 2 | none | 1.00 | 40.29 | 23.11 | 21.08 | — | — | — | — | — |
| rareplanes_independent | 0 | none | 1.00 | — | — | — | 95.82 | 81.68 | 78.57 | — | — |
| rareplanes_independent_seed1 | 1 | none | 1.00 | — | — | — | 96.66 | 80.66 | 81.14 | — | — |
| rareplanes_independent_seed2 | 2 | none | 1.00 | — | — | — | 96.88 | 83.38 | 83.55 | — | — |
| shared_conditional_semantics | 0 | none | 1.00 | 53.27 | 38.76 | 34.35 | 96.87 | 77.75 | 79.96 | — | — |
| shared_conditional_semantics_seed1 | 1 | none | 1.00 | 49.25 | 32.84 | 30.98 | 97.10 | 81.66 | 82.27 | — | — |
| shared_conditional_semantics_seed2 | 2 | none | 1.00 | 48.71 | 32.23 | 30.58 | 96.88 | 76.57 | 79.47 | — | — |
| shared_early_mask | 0 | early | 1.00 | 52.64 | 39.17 | 36.26 | 97.10 | 80.38 | 82.69 | 89.68 | 81.29 |
| shared_early_mask_detached_seed0 | 0 | early_detached | 1.00 | 53.60 | 37.93 | 35.15 | 96.81 | 77.93 | 82.23 | 82.84 | 70.70 |
| shared_early_mask_detached_seed1 | 1 | early_detached | 1.00 | 56.94 | 42.64 | 41.28 | 96.92 | 79.83 | 79.35 | 79.97 | 66.63 |
| shared_early_mask_detached_seed2 | 2 | early_detached | 1.00 | 52.07 | 35.05 | 32.85 | 96.94 | 79.04 | 80.64 | 81.93 | 69.40 |
| shared_early_mask_seed1 | 1 | early | 1.00 | 53.94 | 39.96 | 33.74 | 96.35 | 80.56 | 79.96 | 89.51 | 81.01 |
| shared_early_mask_seed2 | 2 | early | 1.00 | 56.97 | 42.46 | 38.31 | 96.90 | 77.83 | 80.31 | 89.59 | 81.14 |
| shared_early_mask_weight05_seed0 | 0 | early | 0.50 | 53.31 | 36.78 | 32.58 | 96.64 | 77.75 | 79.39 | 89.31 | 80.69 |
| shared_full_mask | 0 | full | 1.00 | 48.85 | 34.38 | 29.71 | 96.78 | 79.17 | 80.80 | 87.59 | 77.92 |
| shared_full_mask_seed1 | 1 | full | 1.00 | 62.11 | 41.96 | 38.41 | 96.95 | 80.96 | 82.34 | 87.38 | 77.58 |
| shared_full_mask_seed2 | 2 | full | 1.00 | 52.83 | 36.28 | 31.68 | 96.95 | 81.65 | 79.37 | 87.83 | 78.31 |
| shared_layer1_mask_seed0 | 0 | layer1 | 1.00 | 55.14 | 38.07 | 36.60 | 96.39 | 77.41 | 79.57 | 88.96 | 80.12 |
| shared_layer1_mask_seed1 | 1 | layer1 | 1.00 | 59.77 | 45.00 | 44.11 | 96.19 | 76.23 | 78.15 | 88.95 | 80.10 |
| shared_layer1_mask_seed2 | 2 | layer1 | 1.00 | 56.52 | 39.40 | 35.40 | 95.74 | 81.35 | 80.14 | 88.89 | 79.99 |
| shared_layer3_mask_seed0 | 0 | layer3 | 1.00 | 58.78 | 45.61 | 42.45 | 96.49 | 78.22 | 80.43 | 88.59 | 79.52 |
| shared_layer3_mask_seed1 | 1 | layer3 | 1.00 | 57.63 | 43.66 | 39.56 | 96.96 | 80.78 | 83.16 | 88.94 | 80.09 |
| shared_layer3_mask_seed2 | 2 | layer3 | 1.00 | 54.99 | 41.94 | 39.20 | 96.41 | 81.36 | 80.64 | 88.95 | 80.10 |

## Three-seed summary

Values are mean ± sample standard deviation in percentage points.

| Family | FineAir acc. | FineAir macro R | FineAir macro F1 | RarePlanes acc. | RarePlanes macro R | RarePlanes macro F1 | DOTA Dice/F1 | DOTA IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fineair_independent | 44.69 ± 3.95 | 30.39 ± 6.76 | 28.10 ± 7.05 | — | — | — | — | — |
| rareplanes_independent | — | — | — | 96.45 ± 0.56 | 81.91 ± 1.38 | 81.09 ± 2.49 | — | — |
| dota_mask_independent | — | — | — | — | — | — | 89.00 ± 0.08 | 80.18 ± 0.12 |
| dota_mask_independent_early | — | — | — | — | — | — | 89.79 ± 0.05 | 81.47 ± 0.08 |
| shared_conditional_semantics | 50.41 ± 2.50 | 34.61 ± 3.61 | 31.97 ± 2.07 | 96.95 ± 0.13 | 78.66 ± 2.67 | 80.57 ± 1.49 | — | — |
| shared_layer1_mask | 57.14 ± 2.38 | 40.83 ± 3.68 | 38.70 ± 4.72 | 96.11 ± 0.33 | 78.33 ± 2.68 | 79.29 ± 1.02 | 88.93 ± 0.04 | 80.07 ± 0.07 |
| shared_early_mask | 54.52 ± 2.22 | 40.53 ± 1.72 | 36.10 ± 2.29 | 96.78 ± 0.39 | 79.59 ± 1.53 | 80.98 ± 1.49 | 89.59 ± 0.08 | 81.15 ± 0.14 |
| shared_layer3_mask | 57.13 ± 1.94 | 43.74 ± 1.84 | 40.41 ± 1.78 | 96.62 ± 0.30 | 80.12 ± 1.67 | 81.41 ± 1.52 | 88.83 ± 0.21 | 79.90 ± 0.33 |
| shared_full_mask | 54.60 ± 6.80 | 37.54 ± 3.94 | 33.27 ± 4.56 | 96.89 ± 0.09 | 80.59 ± 1.27 | 80.84 ± 1.49 | 87.60 ± 0.23 | 77.94 ± 0.36 |
| shared_early_mask_detached | 54.21 ± 2.49 | 38.54 ± 3.83 | 36.43 ± 4.36 | 96.89 ± 0.07 | 78.93 ± 0.95 | 80.74 ± 1.44 | 81.58 ± 1.47 | 68.91 ± 2.08 |
