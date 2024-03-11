import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from .model_parts import CombinationModule
from . import resnet

class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv, resnet_type = "101"):
        super(CTRBOX, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        if resnet_type == "34":
            channels = [3, 16, 64, 128, 512, 1024]
            self.base_network = resnet.resnet34(pretrained=pretrained)
            self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
            self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
            self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
            # for segmentation head, we follow the u-net concept and also have a combination module
            # for the last upconvolution layer
            if 'masks' in heads:
                self.dec_seg1 = CombinationModule(64, 64, batch_norm=True)
                self.dec_seq2 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
        else:
            channels = [3, 64, 256, 512, 1024, 2048]
            self.base_network = resnet.resnet101(pretrained=pretrained)
            self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
            self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
            self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
            if 'masks' in heads:
                self.dec_seg1 = CombinationModule(256, 64, batch_norm=True)
                self.dec_seq2 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )

        self.l1 = int(np.log2(down_ratio))
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'obboxes_params' or head == 'hbboxes_params':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            elif head == 'masks':
                fc = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, classes, kernel_size=1, padding=1, bias=True)
                )
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if head[:4] == "cls_":
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        if 'masks' in self.heads:
            seg_combine = self.dec_seg1(c2_combine, x[-5])
            seg_combine = F.interpolate(
                seg_combine, x[-6].shape[2:], mode='bilinear', align_corners=False
            )
            seg_combine = self.dec_seq2(seg_combine)

        dec_dict = {}
        for head in self.heads:
            if head == 'masks':
                dec_dict[head] = self.__getattr__(head)(seg_combine)
            else:
                dec_dict[head] = self.__getattr__(head)(c2_combine)
            if head[:4] == "cls_" or head == "masks" or head == "obboxes_theta":
                dec_dict[head] = torch.sigmoid(dec_dict[head])

        return dec_dict
