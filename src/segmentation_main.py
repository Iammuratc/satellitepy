from torchvision.transforms import Compose
# import torch
import matplotlib.pyplot as plt

from transforms import Normalize, ToTensor, AddAxis
import models.models as models 
from settings import SettingsSegmentation
from dataset.dataset import DatasetSegmentation 
from classifier.segmentation import ClassifierSegmentation
from utilities import Utilities 
from patch.segmentation import SegmentationPatch

update=True
exp_no = 11
init_features=64
patch_config='original'
dataset_parts = ['train','val']
save_patches = False
exp_name = f'UNet (init_features={init_features}'
output_image='contours' # 'mask'
epochs=50
batch_size=10
learning_rate=1e-3

settings_segmentation = SettingsSegmentation(
											dataset_name='DOTA',
											save_patches=save_patches,
						                    exp_no=exp_no,
						                    exp_name=exp_name,
						                    patience=10,
						                    epochs=epochs,
						                    batch_size=batch_size,
						                    split_ratio=[0.8,0,0.2],
						                    output_image=output_image,
						                    patch_config=patch_config,
						                    bbox_rotation='clockwise',
						                    update=update,
											model_name='UNet',
											learning_rate=learning_rate,
											init_features=init_features,
                                            patch_size=128)()
# print(settings_segmentation)
utils = Utilities(settings_segmentation)

### SAVE PATCHES
if utils.settings['dataset']['save_patches']:
	for dataset_part in dataset_parts:
		segmentation_patch = SegmentationPatch(utils,dataset_part)
		segmentation_patch.get_patches(save=True,plot=False)

### DATASET
dataset = {dataset_part:DatasetSegmentation(
											utils=utils,
											dataset_part=dataset_part,
											transform=Compose([ToTensor(),Normalize(task='segmentation'),AddAxis()])
											) 
											for dataset_part in dataset_parts}
## CLASSIFIER
classifier = ClassifierSegmentation(utils,dataset)

## TRAIN MODEL
# classifier.train()

# ### TRAIN INSTANCE LENGTH
# print(len(dataset))

### PLOT IMAGES
# classifier.plot_images(dataset_part='val')

### CHECK LOADER
# sample = next(loader)

### CHECK SAMPLE HISTOGRAMS
# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(2)
# ax[0].hist(sample['image'].flatten())
# ax[1].hist(sample['label'].flatten())
# plt.show()

# print(sample['image'].shape)
# print(sample['label'].shape)

### MODEL OUTPUT
# model=utils.get_model()
# loader = classifier.get_loader(classifier.dataset['val'],batch_size=1,shuffle=False)
# for sample in loader:
# # 	# outputs = model(sample['image'])
# 	fig,ax = plt.subplots(2)
# 	ax[0].imshow(sample['image'][0])
# 	ax[1].imshow(sample['label'][0])
# 	plt.show()
	# break