from torchvision.transforms import Compose
import random
import torch
# import numpy as np

from recognition import Recognition
from settings import Settings
from models import Custom_0
from transforms import ToTensor, Normalize
from dataset import RecognitionDataset
from classifier import Classifier
# TODO: Store images if they do not exist (e.g., patches with size 32)


### MODEL DEFINITION
exp_no = 0
model_name = 'custom_0'

# TRAINING HYPERPARAMETERS
patch_size=128
batch_size=20
epochs=50
update_settings = True

settings = Settings(model_name=model_name,
                    exp_no=exp_no,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    hot_encoding=True,
                    update=update_settings)()

classifier = Classifier(settings)

### TRAIN
# classifier.train(patience=10)
### TEST
classifier.get_conf_mat(dataset_part='val',save=False,plot=True)

## CHECK DATASET
# print(len(recognition_dataset))
# ind = random.randint(0,len(recognition_dataset)-1)
# sample = recognition_dataset[ind]
# print(sample['label'])

### CHECK MODEL OUTPUT
# dataset = classifier.get_dataset('train')
# loader = classifier.get_loader(dataset)
# model = Custom_0()

# for i,data in enumerate(loader):
#     outputs = model(data['image'])
#     print(outputs)

# #     # print(torch.max(outputs,dim=0))
#     logsoftmax = model.logsoftmax(outputs)
#     print(torch.argmax(logsoftmax,dim=1))
#     print(data['label'])
# # #     print(outputs.shape)
#     break


