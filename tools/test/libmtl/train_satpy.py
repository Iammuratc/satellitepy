import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import json
from satellitepy.data.dataset.mtl_dataset import MTLDataset

from LibMTL import Trainer
from LibMTL.model import resnet18
from LibMTL.metrics import AccMetric
from LibMTL.loss import CELoss
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--satpy_config_path', help='Path to the satellitepy configuration (json file)')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()
    
def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    with open(params.satpy_config_path, "r") as f:
        satpy_cfg = json.load(f)
    
    train_set = MTLDataset(satpy_cfg["train"])
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=params.train_bs,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: x
    )

    test_set = MTLDataset(satpy_cfg["test"])
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=params.test_bs,
        pin_memory=True,
        collate_fn=lambda x: x
    )
    
    # define tasks
    task_dict = {'classification': {'metrics':['Acc'], 
                              'metrics_fn': AccMetric(),
                              'loss_fn': CELoss(),
                              'weight': [1]}, 
                 }
    
    # define encoder and decoders
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            hidden_dim = 512
            self.resnet_network = resnet18(pretrained=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

            # initialization
            self.hidden_layer[0].weight.data.normal_(0, 0.005)
            self.hidden_layer[0].bias.data.fill_(0.1)
            
        def forward(self, inputs):
            out = self.resnet_network(inputs)
            out = torch.flatten(self.avgpool(out), 1)
            out = self.hidden_layer(out)
            return out

    decoders = nn.ModuleDict({task: nn.Linear(512, 18) for task in list(task_dict.keys())})
    
    SatpyTrainer = Trainer(task_dict=task_dict, 
                          weighting=weighting_method.__dict__[params.weighting], 
                          architecture=architecture_method.__dict__[params.arch], 
                          encoder_class=Encoder, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          satpy_input=True,
                          **kwargs)
    SatpyTrainer.train(train_dataloader, test_dataloader, params.epochs)
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
