## Satellitepy

### Requirements

Install the satellitepy repository before running this example.

```shell
cd [satellitepy_path]
pip install -e .
```

### Run a Model

The script `train_satpy.py` is the main file for training and evaluating an MTL model together with satellitepy. A set of command-line arguments is provided to allow users to adjust the training configuration.

Some important arguments are described as follows.

- `weighting`: The weighting strategy. Refer to [here](../../LibMTL#supported-algorithms).
- `arch`: The MTL architecture. Refer to [here](../../LibMTL#supported-algorithms).
- `gpu_id`: The id of gpu. The default value is '0'.
- `seed`: The random seed for reproducibility. The default value is 0.
- `scheduler`: The type of the learning rate scheduler. We recommend to use 'step' here.
- `optim`: The type of the optimizer. We recommend to use 'adam' here.
- `dataset_paths`: The paths of the satellitepy datasets that shall be used.
- `aug`: If `True`, the model is trained with a data augmentation.
- `train_bs`: The batch size of training data. The default value is 8.
- `test_bs`: The batch size of test data. The default value is 8.

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python train_satpy.py -h
```

If you understand those command-line arguments, you can train an MTL model by executing the following command.

```shell
python train_satpy.py --weighting WEIGHTING --arch ARCH --dataset_paths PATH/dataset_1 PATH/dataset_2 --gpu_id GPU_ID --scheduler step
```
