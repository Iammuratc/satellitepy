# Evaluate a pretrained model

**The pretrained model can be shared on request.** The model is pretrained on all data and tasks mentioned in [README](../README.md), and fine-tuned on the training subset of Fair1M. You can follow the steps below to reproduce the results in the table above.

Change your directory to the project directory:

```
cd <satellitepy-dir>
```

The command below will store the predictions of the pretrained model with the corresponding ground truth. You can find more details of the arguments in the script help.

```
python3 tools/test/bbavector/test_original_image.py  --weights-path <weights-dir/CGC_pretrained_on_all_finetuned_on_fair1m.pth --num-workers 0 --input-h 600 --input-w 600 --device cuda:0 --in-image-folder <fair1m-dir>/val/images --in-label-folder <fair1m-dir>/val/labels --K 1000 --out-folder <satellitepy-dir>/test_eval --in-label-format fair1m --patch-size 600 --patch-overlap 100 --tasks hbboxes obboxes masks coarse-class fine-class very-fine-class role difficulty attributes_engines_no-engines attributes_engines_propulsion attributes_fuselage_canards attributes_fuselage_length attributes_wings_wing-span attributes_wings_wing-shape attributes_wings_wing-position attributes_tail_no-tail-fins
```

The prediction-ground truth files will be stored under `<satellitepy-dir>/test_eval/results`, which is called `<result-dir>` from now on. The AP and mAP metrics can be calculated by using the predictions:

```
python3 tools/evaluate/map_of_bbavector_results.py --in-result-folder <result-dir>/predictions --task coarse-class --instance-names airplane,ship,vehicle --out-folder <result-dir>
```

The predictions can also be visualized on the images by using the following command:

```
python tools/data/display_results_on_images.py --in-image-dir <fair1m-dir>/val/images --in-result-dir <result-dir>/predictions --in-mask-dir <result-dir>/masks --out-dir <result-dir>/visuals
 ```