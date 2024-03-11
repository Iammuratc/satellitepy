# Satellitepy
Satellitepy is a python module that focuses on fine-grained object recognition in high resolution optical satellite images. Our target objects are airplane, vessel, vehicle and helicopter. It provides handy tools to deal with many challenges that may rise during object recognition steps.

One of the advantages of Satellitepy is to train a DL model, i.e., MTL-BBAVector, by using multi-task learning, e.g, classification, detection and segmentation. There are 16 tasks in our pipeline. Satellitepy allows each task to be handled individually and/or jointly.

We are trying to keep the code and the documentation up-to-date, so you can use, deploy and develop satellitepy. Feel free to create an issue for your problems/bugs.

# Installation

Please follow the steps in [the installation manual](docs/installing_satellitepy.md) to install satellitepy.

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

You can download a pretrained model from this [link](pretrained_model_link_goes_here). This model is pretrained on all data and tasks, and fine-tuned on Fair1M. You can download [Fair1M](link_to_fair1m) here. You can follow the commands below to reproduce the results in the table above.

 ```
 python3 tools/evaluate.py --in-image-folder ............
 ```


You can also visualize the predictions by the following command:


 ```
 python3 tools/display_results.py --in-image-folder ............
 ```

