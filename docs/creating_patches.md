# What are Patches?

Image patches (or patches in short) are small parts of original images. Since deep learning models moslty need a fixed size of input, e.g., 512 $\times$ 512, 1024 $\times$ 1024, patches are created to unify the image size.

# Downloading datasets
Most well-known satellite imagery datasets for object detection are supported by satellitepy. You can download the datasets by using the scripts in [the folder](tools/data/download). 

# Creating Patches in satellitepy

After you download the datasets, e.g., Fair1M, you can use the following command to create patches by using satellitepy.

```
python3 tools/data/create_patches.py --patch-size 512 --in-image-folder <fair1m-image-folder-path> --in-label-folder <fair1m-label-folder-path> --in-label-format fair1m --out-folder <your-output-folder> --patch-overlap 100
```

Further explanations for the arguments can be found below.
* `--in-image-folder`: Input image folder. This folder consists of image files, for example, .tif files for Fair1M.
* `--in-label-folder`: Input label folder. This folder consists of label files, for example, .xml files for Fair1M.
* `--in-label-format`: Input label format. Since each dataset has its own label format, this should be adjusted accordingly. All label formats can be found in `satellitepy.data.labels`.
* `--out-folder`: Output folder. The patches will be stored in this folder.
* `--patch-overlap`: Stride. The neighboring patches will overlap for <patch-overlap> pixels . This is needed, so each object is present in patches without any cutoff part.
There are many more arguments such as `truncated-object-thr` and `log-path`. You can find all arguments in [the script](tools/data/create_patches.py).

## Rebuilding all original datasets

The paths and layout notes in `docs/orig_data_paths.csv` can be processed in one
reproducible run. The default is 1024 x 1024 patches, 100-pixel overlap, seed 42,
and an original-image-level 80/10/10 split. Splitting before patching prevents
neighboring patches from the same source image leaking between subsets.

```bash
python tools/data/create_original_dataset_patches.py \
  --data-root /data \
  --out-folder /patches
```

Use `--dry-run` to list matched source counts without reading images, or
`--datasets vedai dota` to process only selected datasets. Pass
`--keep-official-splits` when exact published splits are needed instead of the
uniform 80/10/10 split.

Only `vehicle`, `ship`, and `airplane` annotations are retained. The ISPRS
Potsdam/Vaihingen segmentation images are converted to masks and horizontal and
oriented boxes around each vehicle component's convex hull.

VEDAI is written twice using identical source splits:

* `vedai/rgb`: three-channel patches, usable by the default model.
* `vedai/rgbir`: four-channel RGB+IR patches. Train BBAVector with
  `--input-channels 4`; omit the option (or use `3`) for RGB-only experiments.

# Special Cases

There are a few datasets that require some preprocessing before creating patches:

## Rareplanes (+synthetic)
Annotations are provided in a single file for each dataset part. For efficiency and compatibility, the annotations need to be split into one file for each image. Use [split_rareplanes_labels.py](tools/data/split_rareplanes_labels.py).

## Xiew
Similar to Rareplanes, the annotations are provided in a single .geojson file. Split them using [split_xview_into_satellitepy_labels.py](tools/data/split_xview_into_satellitepy_labels.py). Note that the resulting label files are already in satellitepy format, so 'satellitepy' must be used as `label-format`.

## Shipnet and DIOR
All dataset parts are provided in a combined directory, with text files specifying the splits. Use [separate_dataset_parts.py](tools/data/separate_dataset_parts.py) to separate them.
