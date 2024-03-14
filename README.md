# Satellitepy
Satellitepy is a python module that focuses on fine-grained object recognition in high resolution optical satellite images. Our target objects are airplane, vessel, vehicle and helicopter. It provides handy tools to deal with many challenges that may rise during object recognition steps.

One of the advantages of Satellitepy is to train a DL model, i.e., MTL-BBAVector, by using multi-task learning, e.g, classification, detection and segmentation. There are 16 tasks in our pipeline. Satellitepy allows each task to be handled individually and/or jointly.

We keep the code and the documentation up-to-date, so you can use, deploy and develop satellitepy anytime. Feel free to create an issue for your problems/bugs during the usage of satellitepy.

# Datasets
We have trained/evaluated our results in the following datasets: DOTA (w. iSaid), Fair1M, XView, Rarepl. (Real+Synthetic), VEDAI, VHR-10, ShipRSImageNet, UCAS-AOD, DIOR and Potsdam.

## Tasks
We merge and harmonize the tasks in each dataset. The table shows the task avalability for each dataset.

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
The evaluations of fine-tuned MTL-BBAVector models on Fair1M can be found below.
 

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

You can download a pretrained model from this [link](pretrained_model_link_goes_here)*. This model is pretrained on all data and tasks, and fine-tuned on Fair1M. You can download [Fair1M](link_to_fair1m)* here. You can follow the commands below to reproduce the results in the table above.

Change your directory to the project directory:

```
cd <satellitepy-dir>
```

The command below will store the predictions of the pretrained model with the corresponding ground truth, and evaluate the predictions. Please find the details of the arguments in the script help.

```
python tools/test_and_eval_on_original_images.py --weights-path <weights-dir/CGC_pretrained_on_all_finetuned_on_fair1m.pth --tasks hbboxes obboxes masks coarse-class fine-class very-fine-class role difficulty attributes_engines_no-engines attributes_engines_propulsion attributes_fuselage_canards attributes_fuselage_length attributes_wings_wing-span attributes_wings_wing-shape attributes_wings_wing-position attributes_tail_no-tail-fins --target-task coarse-class --in-image-folder <fair1m-dir>/val/images --in-label-folder <fair1m-dir>/val/bounding_boxes --in-label-format fair1m --out-folder <satellitepy-dir>/test_eval --coarse-class-instance-names 'airplane,ship,vehicle'
```

The prediction-ground truth, i.e., result, files will be stored under `<satellitepy-dir>/test_eval/results`, which will be called `<result-dir>` from now on. The result files (`<result-dir>/result_labels`) can be also employed to visualize the predictions on the images by using the following command:

 ```
python tools/data/display_results_on_images.py --in-image-dir <fair1m-dir>/val/images --in-result-dir <result-dir>/result_labels --in-mask-dir <result-dir>/result_labels/result_masks/ --out-dir <result-dir>/result_labels/all_results_on_images
 ```

Note: Please let me know if you can not download Fair1M and the model weights. Unfortunately, it is not allowed to store big data longer than 15 days on our university servers. We try to keep the data up-to-date, however, there might be times when the data is not uploaded.