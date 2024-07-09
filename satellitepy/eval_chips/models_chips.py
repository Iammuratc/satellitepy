import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34, EfficientNet_B3_Weights, \
    efficientnet_b3


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
