from torchvision.transforms import Compose
import random
import torch
# import numpy as np

from settings import SettingsRecognition
from models.models import Custom_0
from transforms import ToTensor, Normalize
# from dataset.dataset import DatasetRecognition
from classifier.recognition import ClassifierRecognition
from utilities import Utilities

# TODO: Store images if they do not exist (e.g., patches with size 32)
#   - Pass settings utils to classifier, and remove the methods like get_model

# EXPERIMENT DEFINITION
model_name = 'custom_0'
# "orthogonal_patch" #"orthogonal_zoomed_patch" , "original_patch"
patch_config = "orthogonal_zoomed_patch"
update = False  # if False, ignore all the parameters defined here, and read the existing settings file
exp_name = f'{model_name}_{patch_config}'

# DATA
dataset_name = 'fair1m'
dataset_parts = ['train', 'val']
save_patches = False
# MODEL DEFINITION

# TRAINING HYPERPARAMETERS
patch_size = 128
batch_size = 20
epochs = 50
split_ratio = [0.8, 0.1, 0.1]
# class_weight=[0.5,1,2,1,1,1,2,1,1,1]
class_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


settings = SettingsRecognition(model_name=model_name,
                               dataset_name=dataset_name,
                               exp_no=exp_no,
                               exp_name=exp_name,
                               patch_size=patch_size,
                               batch_size=batch_size,
                               epochs=epochs,
                               hot_encoding=True,
                               patch_config=patch_config,
                               split_ratio=split_ratio,
                               class_weight=class_weight,
                               update=update)()

# UTILS
utils = Utilities(settings)

# DATASET
dataset = utils.get_dataset(dataset_parts, task='recognition')
# PATCH
# from patch.recognition import RecognitionPatch
# patch = RecognitionPatch(settings,dataset_part='train')
# patch.plot_all_bboxes_on_base_image("/home/murat/Projects/airplane_recognition/data/Gaofen/train/images/12.tif")
# print(settings)

classifier = ClassifierRecognition(utils, dataset)

# ### TRAIN
classifier.train(patience=10)

# ### PLOTTING INSTANCE IMAGES
# classifier.plot_images(instances=['Boeing737','A220'],dataset_part='test',save=True,plot=False)
# classifier.plot_images(instances=['A220','A220'],dataset_part='test',save=True,plot=False)
# classifier.plot_images(instances=['A220','Boeing737'],dataset_part='test',save=True,plot=False)
# classifier.plot_images(instances=['Boeing737','Boeing737'],dataset_part='test',save=True,plot=False)


# CONFUSION MATRIX
# classifier.plot_conf_mat(dataset_part='train',save=True,plot=False)
# classifier.plot_conf_mat(dataset_part='val',save=True,plot=False)
# classifier.get_conf_mat(dataset_part='test',show_false_images=True,save=False,plot=False)


# CHECK MODEL OUTPUT
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