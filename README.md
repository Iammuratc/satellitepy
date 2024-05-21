# Satellitepy
Satellitepy is a python module that focuses on fine-grained object recognition in high resolution optical satellite images. Our target objects are airplane, vessel, vehicle and helicopter. It provides handy tools to deal with many challenges that may rise during object recognition steps.

One of the advantages of Satellitepy is to train a DL model, i.e., MTL-BBAVector, by using multi-task learning, e.g, classification, detection and segmentation. There are 16 tasks in our pipeline. Satellitepy allows each task to be handled individually and/or jointly.

We keep the code and the documentation up-to-date, and resolve the issues as fast as possible, so you can use, deploy and develop satellitepy anytime. Feel free to create an issue for your problems/bugs during the usage of satellitepy.

# Datasets
We have trained/evaluated our models with the following datasets: DOTA (w. iSaid), Fair1M, XView, Rarepl. (Real+Synthetic), VEDAI, VHR-10, ShipRSImageNet, UCAS-AOD, DIOR and Potsdam.

## Tasks
We merge and harmonize various tasks in each dataset. The table shows the task avalability for each dataset.

 Dataset            | HBB | OBB | CGC | Role | FGC | FtGC | Mask | Diff. | Attr. | Res. (m)
 | ---------------- | --- | --- | --- | ---- | --- | ---- | ---- | ----- | ----- | ------- | 
 DOTA (w. iSaid)    | X | X | X  | x | -        | -     | X         | X   | -        | 0.5 |
 Fair1M             | X | X | X   | x  | x  | -    | -         | -        | -        | 0.8 |
 XView              | X  | - | X | x | X   | -    | - | -        | -        | 0.3 |
 Rarepl. (Real)     | X | X | X   | X   | - | -    | - | - | X   | 0.31 |
 Rarepl. (Synth.)   | X | X | X | X   | X   | X  | X   | -        | -   | 0.31 |
 VEDAI              | X | X | X   | -        | X   | -    | - | -        | - | 0.125 |
 VHR-10             | X | - | X | -        | -        | -    | - | - | - | 0.5 - 2 |
 ShipRSImageNet     | X | X | X   | X   | X   | x    | -    | X        | -        | 0.12 - 6 |
 UCAS-AOD           | X | X | X   | -        | -        | -    | -               | -        | -        | 0.5 |
 DIOR               | X  | - | X   | - | - | -    | - |  - | - | 0.5-30 |
 Potsdam            | X | - | X        | x  | -        | -    | X   | - | - | 0.5 |

where,
HBB: Horizontal bounding box, OBB: Oriented bounding box, CGC: Coarse-grained class, FGC: Fine-grained class, FtGC: Finest-grained class, Diff.:Difficulty, Attr.: Attributes, Res.: Spatial Resolution

# Results
The evaluations of fine-tuned MTL-BBAVector models on the test subset of Fair1M can be found below.
 

 Model                                         | CGC   | Role  | FGC  
 | ------------------------------------------- | ----- | ----- | ----- |
 RoI Transformer                               | 0.791 | 0.350 | 0.148 | 
 Rotated RCNN                                  | 0.789 | 0.441 | 0.115 | 
 BBAVector                                     | 0.834 | 0.542 | **0.163** | 
 MTL-BBAVector (not pretr.)                    | 0.834 | 0.545 | 0.080 | 
 MTL-BBAVector (pretr. on single task)         | **0.867** | **0.573** | 0.088 | 
 MTL-BBAVector (pretr. on all tasks)           | 0.862 | 0.401 | 0.088 | 
 MTL-BBAVector (pretr. on best combination)    | 0.860 | **0.573** | 0.085 |

# How to use

## Installation

Please follow the steps in [the installation manual](docs/installing_satellitepy.md) to install satellitepy.

## Evaluate a pretrained model

The pretrained model and the test subset of Fair1M can be downloaded on request*. The model is pretrained on all data and tasks mentioned in the submitted paper, and fine-tuned on the training subset of Fair1M. You can follow the steps below to reproduce the results in the table above.

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

<!-- Note: Please let me know if you are ready to download the Fair1M dataset and the model weights. They will be uploaded when requested, because, unfortunately, it is not allowed to store big data longer than 15 days on our university servers. -->