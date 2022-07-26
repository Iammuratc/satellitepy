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
exp_no = 2
model_name = 'custom_0'
update_settings = False

# TRAINING HYPERPARAMETERS
patch_size=128
batch_size=20
epochs=50
patch_config = "orthogonal_zoomed_patch"#"orthogonal_patch" #"orthogonal_zoomed_patch" , "original_patch"
merge_and_split_data=True
split_ratio=[0.8,0.1,0.1]
class_weight=[0.5,1,2,1,1,1,2,1,1,1]


settings = Settings(model_name=model_name,
                    exp_no=exp_no,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    hot_encoding=True,
                    patch_config=patch_config,
                    merge_and_split_data=True,
                    split_ratio=split_ratio,
                    class_weight=class_weight,
                    update=update_settings)()

# print(settings['training']['class_weight'])
classifier = Classifier(settings)

### TRAIN
# classifier.train(patience=10)
### TEST
classifier.get_conf_mat(dataset_part='test',save=True,plot=True)

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


