# Training MTL-BBAVector model
In satellitepy, you can train a key-point based detection model, called MTL-BBAVector. This model could be trained on multiple tasks (e.g., bounding box detection, classification, segmentation) simultaneously.

# Scripts
You can use the below script to train a MTL-BBAVector model.

```
python3 tools/train/bbavector.py --train-image-folder <train-image-patches-folder-path> --train-label-folder <train-label-patches-folder-path> --train-label-format <train-label-format> --valid-image-folder <validation-image-patches-folder-path> --valid-label-folder <validation-label-patches-folder-path> --valid-label-format <validation-label-format>  --out-folder <output-folder> --tasks <your-task-list> 
```

The explanations for some arguments can be found below:
* `--train-image-folder`: Input image folder for training. This folder consists of image files for training the model.
* `--train-label-folder`: Input label folder for training. This folder consists of label files for training the model.
* `--train-label-format`: Input label format for training. Since each dataset has its own label format, this should be adjusted accordingly. All label formats can be found in `satellitepy.data.labels`.
* `--valid-image-folder`: Input image folder for validation. This folder consists of image files for validating the model.
* `--valid-label-folder`: Input label folder for validation. This folder consists of label files for validating the model.
* `--valid-label-format`: Input label format for validation. Since each dataset has its own label format, this should be adjusted accordingly. All label formats can be found in `satellitepy.data.labels`.
* `--out-folder`: Output folder. The prediction results (.json) for the validation images will be stored in this folder.
* `--tasks`: Task list. The model will be trained for <tasks> jointly. For example, segmentation from DOTA/iSaid and fine-grained classification from Fair1M can be jointly trained within the MTL-BBAVector model.

There are many more arguments such as `batch-size` and `init-lr`. You can find all arguments in [the script](tools/train/bbavector.py) with their explanations, or you can run the help command in a terminal:

```
python3 tools/train/bbavector.py --help
```

Validation is optional, but its image folder, label folder, and label format must be supplied together.
Pass task names as separate arguments, for example `--tasks obboxes coarse-class role`.

## Docker

Build the CPU image from the repository root:

```bash
docker build -t satellitepy .
```

For the local dataset layout recorded in `docs/orig_data_paths.csv`, Compose mounts
`/mnt/2tb-0/satellitepy/data` read-only at `/data`. The datasets are therefore available
inside the container at these paths:

| Dataset | Container path |
| --- | --- |
| DIOR | `/data/DIOR/split` |
| DOTA | `/data/DOTA` |
| Fair1M | `/data/fair1m` |
| Potsdam/Vaihingen | `/data/Potsdam/archive/ISPRS_semantic_labeling_Vaihingen` |
| RarePlanes | `/data/Rareplanes/train` |
| VEDAI | `/data/VEDAI` |
| VHR-10 | `/data/VHR-10_dataset_coco/VHR-10-NWPU` |
| Ship Image Net | `/data/ship_image_net/split` |
| UCAS-AOD | `/data/ucas_aod` |
| xView | `/data/xview` |
| FineAir | `/data/FineAir/FR24_dataset` |

Start a shell with all datasets mounted and `./outputs` mounted at `/output`:

```bash
mkdir -p outputs
docker compose run --rm satellitepy bash
```

Generated dataset patches should be written to `/patches` in the container. This is
mounted from `/mnt/2tb-1/satellitepy/patches` on the host. Override the location with
`SATELLITEPY_PATCHES_ROOT` when necessary.

The image also installs the Codex CLI. The repository is bind-mounted into the
container, while Codex configuration is retained in the `codex-home` Docker volume.
Authenticate interactively after the first container start; credentials are not stored
in the image:

```bash
codex login
codex --version
```

The Compose service requests all available NVIDIA GPUs and builds with CUDA-enabled
PyTorch by default. This requires the NVIDIA driver and NVIDIA Container Toolkit on
the host. Verify GPU access with:

```bash
docker compose run --rm satellitepy python -c "import torch; print(torch.cuda.is_available())"
```

To build a CPU-only image instead, override the wheel index and remove or comment out
`gpus: all` in `compose.yaml`:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu docker compose build
```

If the data root changes, override it without editing the Compose file:

```bash
SATELLITEPY_DATA_ROOT=/another/data/root docker compose run --rm satellitepy bash
```

FineAir is mounted separately because its default host location is on the
second data volume. Override that location without editing Compose with
`SATELLITEPY_FINEAIR_ROOT=/another/FineAir/root`.

Mount datasets and outputs instead of copying them into the image:

```bash
docker run --rm -it \
  -v /absolute/path/to/data:/data:ro \
  -v /absolute/path/to/output:/output \
  satellitepy python tools/train/bbavector.py \
  --train-image-folder /data/train/images \
  --train-label-folder /data/train/labels \
  --train-label-format satellitepy \
  --out-folder /output \
  --tasks obboxes coarse-class
```

For a CUDA build, set `TORCH_INDEX_URL` to the matching PyTorch wheel index and run the
container with the NVIDIA Container Toolkit, for example:

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 -t satellitepy:cuda .
docker run --rm --gpus all satellitepy:cuda python -c "import torch; print(torch.cuda.is_available())"
```
