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