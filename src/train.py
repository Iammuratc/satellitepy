import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
    

from transform import Augmentations, ToTensor
from utilities import show_sample
from dataset import AirplaneDataset
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)


# Dataloader
airplane_dataset = AirplaneDataset(dataset_name='train',patch_size=512,transform=Compose([Augmentations(),ToTensor()]))

dataloader = DataLoader(airplane_dataset, batch_size=4,shuffle=True, num_workers=0)


train_features, train_labels = next(iter(dataloader))

print(train_features.size())
print(train_labels.size())
# show_sample(train_features[0])