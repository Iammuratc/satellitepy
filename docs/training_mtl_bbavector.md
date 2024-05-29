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