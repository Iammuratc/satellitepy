import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from satellitepy.data.dataset.mtl_dataset import MTLDataset
from satellitepy.utils.libmtl.task_target_mapping import build_targets
import torchvision

from LibMTL import Trainer
from LibMTL.model import resnet18
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

    with open(params.satpy_config_path, 'r') as f:
        satpy_cfg = json.load(f)

    train_set = MTLDataset(satpy_cfg['train'], torchvision.transforms.Resize((100, 100), antialias=True))
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=params.train_bs,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_set = MTLDataset(satpy_cfg['test'], torchvision.transforms.Resize((100, 100), antialias=True))
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=params.test_bs,
        pin_memory=True,
    )

    task_dict = build_targets(satpy_cfg['train']['tasks'])

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            hidden_dim = 512
            self.resnet_network = resnet18(pretrained=True)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                      nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

            self.hidden_layer[0].weight.data.normal_(0, 0.005)
            self.hidden_layer[0].bias.data.fill_(0.1)

        def forward(self, inputs):
            out = self.resnet_network(inputs)
            out = torch.flatten(self.avgpool(out), 1)
            out = self.hidden_layer(out)
            return out

    decoders = nn.ModuleDict({
        'attributes_engines_no-engines': nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        ),
        'attributes_engines_propulsion': nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        ),
        'attributes_wings_wing-span': nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Flatten(start_dim=0)
        ),
        'role': nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
    })

    satpy_trainer = Trainer(task_dict=task_dict,
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
    satpy_trainer.train(train_dataloader, test_dataloader, params.epochs)


if __name__ == '__main__':
    params = parse_args(LibMTL_args)
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    main(params)
