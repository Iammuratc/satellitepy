# BBAVector

## Basic installation

Add conda to PATH:
```bash
export PATH="/opt/anaconda3/bin":$PATH
bash --login
```
Create venv and install BBAVectors:
```bash
conda create -n bba python=3.10
conda activate bba
conda install opencv pytorch matplotlib shapely
git clone https://github.com/yijingru/BBAVectors-Oriented-Object-Detection
# Installing Dota Devkit
cd BBAVectors-Oriented-Object-Detection/datasets/
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## Changes
datasets/transforms.py:
Rename all occurences of np.int to int

datasets/hrsc_evaluation_task1.py: Delete `import DOTA_devkit.polyiou`
datasets/DOTA_DEVKIT/ResultMerge_multi_process.py: delete `import DOTA_devkit.polyiou` and comment the following line:
```python
try:
    import dota_utils as util
except:
    rt dota_kit.dota_utils as util
```

## Split DOTA Images
1. Run ImgSplit.py in DOTA_devkit folder
2. The folder examplesplit will be created
3. Create trainval.txt and test.txt and add the content from below

## Run training
```bash
python main.py --data_dir datasets/DOTA_devkit/examplesplit --batch_size 1 --dataset dota --phase train
```

test.txt and trainval.txt
```
P0706__1__0___0
P0706__1__0___158
P0706__1__87___0
P0706__1__87___158
P0770__1__0___0
P0770__1__0___334
P0770__1__586___0
P0770__1__586___334
P1088__1__0___0
P1088__1__0___879
P1088__1__575___0
P1088__1__575___879
P1234__1__0___0
P1234__1__0___1848
P1234__1__0___2772
P1234__1__0___2976
P1234__1__0___924
P1234__1__1848___0
P1234__1__1848___1848
P1234__1__1848___2772
P1234__1__1848___2976
P1234__1__1848___924
P1234__1__2772___0
P1234__1__2772___1848
P1234__1__2772___2772
P1234__1__2772___2976
P1234__1__2772___924
P1234__1__2976___0
P1234__1__2976___1848
P1234__1__2976___2772
P1234__1__2976___2976
P1234__1__2976___924
P1234__1__924___0
P1234__1__924___1848
P1234__1__924___2772
P1234__1__924___2976
P1234__1__924___924
P1888__1__0___0
P2598__1__0___0
P2598__1__0___1030
P2598__1__0___924
P2598__1__1332___0
P2598__1__1332___1030
P2598__1__1332___924
P2598__1__924___0
P2598__1__924___1030
P2598__1__924___924
P2709__1__0___0
P2709__1__0___1390
P2709__1__0___924
P2709__1__1848___0
P2709__1__1848___1390
P2709__1__1848___924
P2709__1__1966___0
P2709__1__1966___1390
P2709__1__1966___924
P2709__1__924___0
P2709__1__924___1390
P2709__1__924___924
```
