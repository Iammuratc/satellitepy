import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
    

from transform import Augmentations, ToTensor
from utilities import show_sample
from dataset import AirplaneDataset
from model import get_yolo


# def my_collate(batch):
#     print(batch[0]['image'].shape)
#     data = [item['image'] for item in batch]
#     target = [item['orthogonal_bboxes'] for item in batch]
#     target = torch.LongTensor(target)

#     return [data, target]
def collate_fn(batch):
    return tuple(zip(*batch))

model = get_yolo(print_summary=True)


# Dataloader
# airplane_dataset = AirplaneDataset(dataset_name='train',patch_size=512,transform=Compose([Augmentations(),ToTensor()]))
# dataloader = DataLoader(airplane_dataset, batch_size=6, shuffle=True, num_workers=1,collate_fn=collate_fn)
# images,targets = next(iter(dataloader))

# print(len(targets))
# print(targets)


