# Training MTL-BBAVector model
In satellitepy, you can train a key-point based detection model, called MTL-BBAVector. This model could be trained on multiple tasks (e.g., bounding box detection, classification, segmentation) simultaneously.

# Scripts
You can use the below script to train a MTL-BBAVector model on several tasks.

```
python3 tools/train/bbavector.py --train-image-folder < --train-label-folder /mnt/2tb-0/satellitepy/patches/all/train/labels/ --train-label-format satellitepy --valid-image-folder /mnt/2tb-0/satellitepy/patches/all/val/images/ --valid-label-folder /mnt/2tb-0/satellitepy/patches/all/val/labels/ --valid-label-format satellitepy  --out-folder exps/bbavector_dota_08_28  --batch-size 2 --ngpus 0 --num-workers 1 --num-epoch 5  --init-lr 5e-4 --tasks hbboxes obboxes coarse-class fine-class role masks 
```