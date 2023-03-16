# Satellitepy
The Satellitepy package is a python module that focuses on fine-grained object recognition in high resolution satellite images. Satellitepy provides handy tools to deal with many challenges that may rise during object recognition steps. 

One of the greatest advantages of Satellitepy is the task separation. As CNN models mostly deal with detection and classification tasks simultaneously, it could be difficult to spot where exactly the low accuracy results from. In Satellitepy, every task is developed individually to handle the overload of multiple tasks for CNN models. The implemented approaches for each task are listed below. One can also found the list of our publications.

We develop an organized pipeline and a well-written documentation such that everyone can use, contribute to and deploy `satellitepy` into their environments.

# Installation

**Clone satellitepy**

`git clone git@github.com:Iammuratc/satellitepy.git`

**Create a virtual environment**

`cd satellitepy`

`python3 -m venv venv_satellitepy`

`source venv_satellitepy/bin/activate`

**Install dependencies**

The installation is only tested on python3.10, please open an issue if you have any problems during any step of the installation.

**Pytorch**

`pip3 install torch torchvision torchaudio`

Please check out [pytorch.org](https://pytorch.org/) for installing the version that is compatible with your environment (CUDA, conda etc.).

**MMRotate**

Our detection models rely on MMRotate. FYI, the installation of mmcv-full might take some time (30 min for me), please be patient. We are working on removing the MMRotate dependency. Please check out [MMRotate installation page](https://mmrotate.readthedocs.io/en/latest/install.html) for more details.

`pip install openmim`

`mim install mmcv-full` 

`mim install mmdet\<3.0.0`

`pip install mmrotate`

**Satellitepy dependencies**

`pip3 install -r requirements.txt`

`pip3 install -e .`

You are ready to work with satellitepy!
# Datasets
We support the following datasets:

- Fair1M
- DOTA
- RarePlanes

# Tasks
## Detection
We use [MMRotate](https://github.com/open-mmlab/mmrotate) to detect oriented bounding boxes (OBB) of coarse-grained objects in satellite images. 

For errors during compiling the mmrotate module, refer to the [troubleshooting](docs/troubleshooting_mmrotate.md) section.

