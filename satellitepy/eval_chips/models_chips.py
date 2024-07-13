import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34, EfficientNet_B3_Weights, \
    efficientnet_b3, ResNet50_Weights, resnet50, Swin_T_Weights, swin_t, ConvNeXt_Small_Weights, convnext_small, \
    ShuffleNet_V2_X1_5_Weights, shufflenet_v2_x1_5


def get_model(model_name):
    if model_name == 'efficientnet_b3':
        weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weights)
    elif model_name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif model_name == 'resnet34':
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == 'swin_t':
        weights = Swin_T_Weights.DEFAULT
        model = swin_t(weights=weights)
    elif model_name == 'convnext_small':
        weights = ConvNeXt_Small_Weights.DEFAULT
        model = convnext_small(weights=weights)
    elif model_name == 'shufflenet_v2_x1_5':
        weights = ShuffleNet_V2_X1_5_Weights.DEFAULT
        model = shufflenet_v2_x1_5(weights=weights)
    else:
        print(f'Model {model_name} not supported')

    return model


class Head(torch.nn.Module):
    def __init__(self, num_classes, model):
        super(Head, self).__init__()

        self.num_classes = num_classes
        self.linear = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.softmax(x, dim=1)

        return x


def get_head(model, num_classes):
    return Head(num_classes=num_classes, model=model)
