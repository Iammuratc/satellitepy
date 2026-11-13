# DOTA-v2 annotation overlay

The mounted `/data/DOTA` directory contains DOTA-v1 images and is read-only.
DOTA-v2 relabelled train and validation annotations are therefore stored as a
non-destructive overlay under `/output/dota_v2_annotations`.

The official DOTA instructions state that DOTA-v1 images were relabelled for
v2 and should be paired with the v2 annotations. The official OneDrive folder
is interactive, so the two annotation archives were obtained from the public
`isaaccorley/dota` mirror of the DOTA release.

| Archive | SHA-256 |
| --- | --- |
| `dotav2.0_annotations_train.tar.gz` | `f142824f6eafef3b1922f7ff1583375e4d59d2a841d4670c312be854652631a6` |
| `dotav2.0_annotations_val.tar.gz` | `66cde2e76f131a55243904884b68ef450a73b1404668001df92bc94dde4a2e25` |

All 1,411 mounted v1 train image IDs and all 458 mounted v1 validation image
IDs have matching DOTA-v2 annotation files. The archives also contain labels
for 419 additional train images and 135 additional validation images whose
corresponding v2 images are not mounted; those labels are retained but are not
used.

For the image IDs available locally, target-object counts change as follows:

| Split | Version | Plane | Ship | Small vehicle | Large vehicle |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | v1 | 8,055 | 28,068 | 26,126 | 16,969 |
| Train | v2 | 8,072 | 32,973 | 126,501 | 22,218 |
| Validation | v1 | 2,531 | 8,960 | 5,438 | 4,387 |
| Validation | v2 | 2,550 | 10,765 | 43,337 | 5,139 |

This is not a metadata-only replacement: DOTA-v2 adds many small-object
annotations, especially small vehicles, so existing patch labels must be
regenerated rather than renamed or copied in place.

`tools/data/create_original_dataset_patches.py` selects the overlay
automatically through its default `--dota-v2-label-root`. Existing DOTA patch
outputs are not silently overwritten; regenerate them explicitly to apply the
new annotations.

The dataset is restricted to academic/non-commercial use under the DOTA terms.
