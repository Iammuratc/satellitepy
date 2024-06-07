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

Please follow the steps in [the installation manual](docs/installing_satellitepy.md) to install satellitepy. After that, you can perform the following:

1) You can [create patches](docs/creating_patches.md) from original images to train/test your models.
2) You can [train a model](docs/training_mtl_bbavector.md) on your [patches](docs/creating_patches.md).
3) You can [evaluate a pretrained model](docs/evaluating_pretrained_model.md) on your original test images. The model will make predictions for patches of the original image and merge the patch predictions automatically.

# Citation

Please cite the following paper if you use satellitepy in your research.

@inproceedings{osswald2023fine,
  title={Fine-Grained Airplane Recognition in Satellite Images based on Task Separation and Orientation Normalization},
  author={Osswald-Cankaya, Murat and Mayer, Helmut},
  booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
  pages={6545--6548},
  year={2023},
  organization={IEEE}
}