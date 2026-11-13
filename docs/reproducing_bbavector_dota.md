# Reproducing BBAVector on DOTA-v1.0

This workflow is separate from the three-class satellitepy dataset. It uses all
15 DOTA-v1.0 categories under the `dota-class` task.

Four-GPU training requires a sufficiently large container shared-memory mount.
The provided `compose.yaml` sets `shm_size: 16gb`; recreate an older running
container after pulling this change. Docker's 64 MiB default is insufficient
for ResNet-101 replication and DataLoader workers.

The paper uses 608 x 608 network input. The authors' released preprocessing is
slightly different from simply making 608-pixel patches: it crops original DOTA
images into 600 x 600 windows with a 100-pixel overlap/gap setting at scales 1.0
and 0.5, and the loader resizes those crops to 608 x 608. The preparation script
reproduces that distinction.

## Prepare trainval and official test

```bash
python tools/data/prepare_dota_bbavector.py \
  --dota-root /data/DOTA \
  --out-folder /patches/DOTA_bbavector
```

`trainval` combines the official labelled train and validation images. Empty
training crops are removed, matching the authors' recommendation. Every official
test crop is retained because test annotations are unavailable.

## Train

The paper reports Adam with learning rate 1.25e-4, random flipping and random
crop scale 0.9--1.1, roughly 40 epochs, and batch size 20 over four GTX 1080 Ti
GPUs. The authors' current repository example uses 80 epochs and batch size 16;
use the paper settings below for the reported 72.32 mAP experiment:

```bash
python tools/train/bbavector.py \
  --train-image-folder /patches/DOTA_bbavector/trainval/images \
  --train-label-folder /patches/DOTA_bbavector/trainval/labels \
  --train-label-format satellitepy \
  --tasks obboxes hbboxes dota-class \
  --target-task dota-class \
  --input-h 608 --input-w 608 \
  --num-epoch 40 --batch-size 20 --ngpus 4 \
  --checkpoint-interval 5 \
  --init-lr 1.25e-4 --augmentation \
  --out-folder /output/bbavector_dota
```

If container NCCL support is unavailable, the equivalent single-GPU setup uses
`--batch-size 5 --ngpus 1 --gradient-accumulation-steps 4`, retaining an
effective optimizer batch of 20 and the original per-device BatchNorm batch.

The starred 75.36 mAP result used batch size 48 on four RTX 6000 GPUs.

## Run official-test inference

Run inference on the prepared two-scale test patches:

```bash
python tools/test/bbavector/test_patch.py \
  --weights-path /output/bbavector_dota/model_no_valid_39.pth \
  --in-image-folder /patches/DOTA_bbavector/test/images \
  --out-folder /output/bbavector_dota_test \
  --tasks obboxes hbboxes dota-class --target-task dota-class \
  --input-h 608 --input-w 608 \
  --conf-thresh 0.01 --K 1000
```

The exporter recognizes the scale and source coordinates encoded in every patch
name, maps detections back to original-image coordinates, combines both scales,
and applies class-wise polygon NMS at IoU 0.1. It then creates the 15
`Task1_<class>.txt` files for the DOTA Task1 evaluation server:

```bash
python tools/evaluate/create_dota_submission.py \
  --result-folder /output/bbavector_dota_test/results/predictions \
  --out-folder /output/bbavector_dota_test/Task1
cd /output/bbavector_dota_test && zip -r Task1.zip Task1
```

The original paper is `docs/Yi_Oriented_Object_Detection_in_Aerial_Images_With_Box_Boundary-Aware_Vectors_WACV_2021_paper.pdf`.
