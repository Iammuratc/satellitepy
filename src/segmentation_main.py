from torchvision.transforms import Compose
from transforms import Normalize, ToTensor
import torch

from unet import UNet
from settings import SettingsSegmentation
from dataset import SegmentationDataset 
from segmentation_classifier import SegmentationClassifier



exp_no = 7
exp_name = 'Initial UNet'
epochs=50
batch_size=10

settings_segmentation = SettingsSegmentation(
											dataset_name='DOTA',
						                    exp_no=exp_no,
						                    exp_name=exp_name,
						                    patience=10,
						                    epochs=epochs,
						                    batch_size=batch_size,
						                    split_ratio=[0.8,0.2,0],
						                    bbox_rotation='clockwise',
						                    update=True,
											model_name='UNet',
                                            patch_size=128)()
# print(settings_segmentation)


dataset_part='train'
## CLASSIFIER
classifier = SegmentationClassifier(settings_segmentation)

## TRAIN MODEL
classifier.train()

# ### TRAIN INSTANCE LENGTH
# print(len(dataset))

### PLOT IMAGES
# classifier.plot_images(dataset_part)

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

## CHECK MODEL
# model = classifier.get_model()
# ### MODEL OUTPUT

# loader = classifier.get_loader(classifier.get_dataset('train'),batch_size=1,shuffle=False)
# for sample in loader:
# 	outputs = model(sample['image'])
# 	print(sample['label'].shape)
# 	print(outputs.shape)
# 	break


# dataset = SegmentationDataset(		settings=settings_segmentation,
# 									dataset_part=dataset_part,
# 									# transform=Compose([Normalize(task='segmentation')]))#,ToTensor()]))
# 									transform=Compose([ToTensor(),Normalize(task='segmentation')]))

# loader = torch.utils.data.DataLoader(dataset, 
#                                     batch_size=1,
#                                     shuffle=False, 
#                                     num_workers=1)      