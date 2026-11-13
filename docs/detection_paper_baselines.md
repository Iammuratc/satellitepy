# Three-seed detector comparison

Test mAP50-95, reported as mean ± sample standard deviation over seeds 0-2. Deltas are seed-paired against the corresponding independent detector.

| Model | Dataset | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | Paired delta ± SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Independent detectors | FineAir | 0.9385 | 0.9266 | 0.9321 | 0.9324 ± 0.0059 | +0.0000 ± 0.0000 |
| Independent detectors | RarePlanes | 0.7826 | 0.7930 | 0.7871 | 0.7876 ± 0.0052 | +0.0000 ± 0.0000 |
| Independent detectors | DOTA-aircraft | 0.8171 | 0.8036 | 0.8095 | 0.8101 ± 0.0068 | +0.0000 ± 0.0000 |
| strict-detached | FineAir | 0.9347 | 0.9365 | 0.9332 | 0.9348 ± 0.0017 | +0.0025 ± 0.0070 |
| strict-detached | RarePlanes | 0.7911 | 0.7907 | 0.7910 | 0.7909 ± 0.0002 | +0.0033 ± 0.0054 |
| strict-detached | DOTA-aircraft | 0.7984 | 0.8165 | 0.8121 | 0.8090 ± 0.0095 | -0.0010 ± 0.0161 |
| DOTA adapter | FineAir | 0.9294 | 0.9379 | 0.9403 | 0.9359 ± 0.0057 | +0.0035 ± 0.0110 |
| DOTA adapter | RarePlanes | 0.7858 | 0.7837 | 0.7917 | 0.7870 ± 0.0041 | -0.0005 ± 0.0077 |
| DOTA adapter | DOTA-aircraft | 0.8150 | 0.8132 | 0.8068 | 0.8117 ± 0.0043 | +0.0016 ± 0.0069 |

With only three seeds, this table supports effect-size and stability comparisons, not a strong claim of statistical significance. Neither shared model consistently improves all three datasets over independent training.
