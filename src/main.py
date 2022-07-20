from torchvision.transforms import Compose
import random
import torch

from recognition import Recognition
from settings import Settings
from models import Custom_0
from transforms import ToTensor, Normalize
from dataset import RecognitionDataset

# TODO: Store images if they do not exist (e.g., patches with size 32)


exp_no = 0
update_settings = True
batch_size=10
patch_size=128
dataset_part = 'train'

settings = Settings(batch_size=batch_size,hot_encoding=True,exp_no=0,patch_size=patch_size,update=update_settings)()
dataset = RecognitionDataset(settings,dataset_part,transform=Compose([ToTensor(),Normalize()]))
loader = torch.utils.data.DataLoader(dataset, batch_size=settings['training']['batch_size'],shuffle=True, num_workers=2)

## CHECK DATASET
# print(len(recognition_dataset))
# ind = random.randint(0,len(recognition_dataset)-1)
# sample = recognition_dataset[ind]
# print(sample['label'])

### CHECK MODEL OUTPUT
model = Custom_0()

for i,data in enumerate(loader):
    outputs = model(data['image'])
    print(outputs.shape)
    break


