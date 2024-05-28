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

# Special Cases

There are a few datasets that require some preprocessing before creating patches:

## Rareplanes (+synthetic)
Annotations are provided in a single file for each dataset part. For efficiency and compatibility, the annotations need to be split into one file for each image. Use [split_rareplanes_labels.py](tools/data/split_rareplanes_labels.py).

## Xiew
Similar to Rareplanes, the annotations are provided in a single .geojson file. Split them using [split_xview_into_satellitepy_labels.py](tools/data/split_xview_into_satellitepy_labels.py). Note that the resulting label files are already in satellitepy format, so 'satellitepy' must be used as `label-format`.

## Shipnet and DIOR
All dataset parts are provided in a combined directory, with text files specifying the splits. Use [separate_dataset_parts.py](tools/data/separate_dataset_parts.py) to separate them.