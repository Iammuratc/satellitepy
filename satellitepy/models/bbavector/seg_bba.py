import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class seg_bba(nn.Module):
    def __init__(self, seg_model, bba_model):
        super(seg_bba, self).__init__()
        self.seg_model = seg_model
        self.bba_model = bba_model

    def forward(self, x):
        x.requires_grad = True
        z = checkpoint(self.seg_model, x)
        z = torch.cat((x, z['masks']), dim=1)
        z = self.bba_model(z)
        return z