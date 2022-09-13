### CREATE PATCHES
from settings import SettingsDetection
from patch.detection import PatchDetection
#512 as patch_size
#patch part only for fair1m, not dota
settings = SettingsDetection(patch_size=256,
                            dataset_name='FAIR1m',label_names=['Boeing787','Boeing737','Boeing747','Boeing787', 'A220', 'A321', 'A330', 'A350', 'ARJ21','other-airplane'])()
# add labels here in the settings

for dataset_part in ['train','test','val']:
    detection_patch = PatchDetection(settings,dataset_part=dataset_part)
    detection_patch.get_patches(save=True,plot=False)    




## EXPERIMENTS
# from detection_classifier import DetectionClassifier
# exp_no = 5
# exp_name = 'First yolov5m detection trial'

# ### PATCH
# patch_size = 512
# split_ratio = [0.8,0.1,0.1]
# batch_size = 10
# epocs = 50

# model_name = 'yolov5m'


# settings = SettingsDetection(update=True,
#                             model_name=model_name,
#                             exp_no=exp_no,
#                             exp_name=exp_name,
#                             patch_size=patch_size,
#                             split_ratio=split_ratio,
#                             batch_size=batch_size,
#                             epochs=epocs)()


# classifier = DetectionClassifier(settings)

# classifier.get_patch_folders()