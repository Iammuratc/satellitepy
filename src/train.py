# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose
    

# from transform import Augmentations, ToTensor
# from utilities import show_sample
# from dataset import AirplaneDataset
# from model import get_yolo


import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
print(model.eval())



