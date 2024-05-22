# Training MTL-BBAVector model
In satellitepy, you can train a key-point based detection model, called MTL-BBAVector. This model could be trained on multiple tasks (e.g., bounding box detection, classification, segmentation) simultaneously.

# Scripts
You can use the below script to train a MTL-BBAVector model.

```
python3 tools/train/bbavector.py --train-image-folder <train-image-patches-folder-path> --train-label-folder <train-label-patches-folder-path> --train-label-format <train-label-format> --valid-image-folder <validation-image-patches-folder-path> --valid-label-folder  <validation-label-patches-folder-path> --valid-label-format <validation-label-format>  --out-folder <output-folder> --tasks <your-task-list> 
```

The explanations for some arguments can be found below:
