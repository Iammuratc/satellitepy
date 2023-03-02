# Satellitepy
The Satellitepy package is a python module that focuses on fine-grained object recognition in high resolution satellite images. Satellitepy provides handy tools to deal with many challenges that may rise during object recognition steps. 

One of the greatest advantages of Satellitepy is the task separation. As CNN models mostly deal with detection and classification tasks simultaneously, it could be difficult to spot where exactly the low accuracy results from. In Satellitepy, every task is developed individually to handle the overload of multiple tasks for CNN models. The implemented approaches for each task are listed below. One can also found the list of our publications.

We develop an organized pipeline and a well-written documentation such that everyone can use, contribute to and deploy `satellitepy` into their environments.

# Installation

`pip install satellitepy`

# Datasets
We support the following datasets:
- Fair1M
- DOTA
- RarePlanes

# Tasks
## Detection
We use [MMRotate](https://github.com/open-mmlab/mmrotate) to detect oriented bounding boxes (OBB) of coarse-grained objects in satellite images. 

### Possible problems
KeyError: 'RotatedRetinaNet is not in the models registry'
### Solution
Add the following lines to your config file:
`custom_imports = dict(
    imports=['mmrotate.models.detectors.rotated_retinanet'],
    allow_failed_imports=False)`

