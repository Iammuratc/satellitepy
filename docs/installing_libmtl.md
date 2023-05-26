# LibMTL

## Basic installation
Add conda to PATH:
```bash
export PATH="/opt/anaconda3/bin":$PATH
bash --login
```
Create venv and install LibMTL:
```bash
conda create -n libmtl python=3.8
conda activate libmtl
pip install torch torchvision numpy==1.20

git clone https://github.com/median-research-group/LibMTL.git

cd LibMTL
pip install -e .
pip install scipy cvxpy
```

## Changes in Code
Make required changes due to syntax changes in newer pytorch version:

In LibMTL/model/resnet.py:
```python
from torchvision.models.utils import load_state_dict_from_url
```
to
```python
from torch.hub import load_state_dict_from_url
```

In LibMTL/trainer.py (row 132):
```python
def _process_data(self, loader):
        try:
            data, label = loader[1].next()
        except:
            loader[1] = iter(loader[0])
            data, label = loader[1].next()
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
```
to
```python
def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_bl>
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
```

## Run training
```bash
cd examples/nyu/
python train_nyu.py --weighting EW --arch HPS --dataset_path /PATH/TO/NYUV2/ --gpu_id 3 --scheduler step
```