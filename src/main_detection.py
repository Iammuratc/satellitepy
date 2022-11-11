# CREATE PATCHES
from src.settings.experiment import SettingsDetection
from src.data.patch.detection import PatchDetection
# from utilities import Utilities
from classifier.detection import ClassifierDetection

# 512 as patch_size
# patch part only for fair1m, not dota

update = True
model_name = 'yolov5m'
exp_name = f"{model_name}_0"
split_ratio = [0.802, 0.1, 0.098]
batch_size = 10
patience = 10
save_patches = False
dataset_name = 'fair1m'
box_corner_threshold = 2
epochs = 50
patch_size = 512
overlap = 24
dataset_parts = ['train', 'val']
label_names = [
    'Boeing787',
    'Boeing737',
    'Boeing747',
    'Boeing787',
    'A220',
    'A321',
    'A330',
    'A350',
    'ARJ21',
    'C919',
    'other-airplane']

settings = SettingsDetection(
    update=update,
    model_name=model_name,
    exp_name=exp_name,
    patch_size=patch_size,
    overlap=overlap,
    split_ratio=split_ratio,
    batch_size=batch_size,
    patience=patience,
    save_patches=save_patches,
    dataset_name=dataset_name,
    box_corner_threshold=box_corner_threshold,
    epochs=epochs,
    label_names=label_names)()
# add labels here in the settings
# UTILITIES
# tools = Utilities(settings)

# SAVE PATCHES
if save_patches:
    for dataset_part in dataset_parts:
        detection_patch = PatchDetection(settings, dataset_part=dataset_part)
        detection_patch.get_patches(save=True, plot=False)

# ## CLASSIFIER
classifier = ClassifierDetection(settings)

classifier.train()
