from settings.dataset import SettingsDataset
from src.data.cutout.utilities import filter_truncated_images
from src.data.tools import normalize_rarePlanes_annotations
from utilities import write_cutouts, convert_my_labels_to_imagenet
### DATASET

dota_settings = SettingsDataset(
    dataset_name='DOTA',
    dataset_parts=['train','val'],
    tasks=['bbox','seg'],
    bbox_rotation='clockwise',
    # project_folder='F:\\working',
    instance_names=['plane'])(),
# print(dota_settings)

fair1m_settings = SettingsDataset(
    dataset_name='fair1m',
    dataset_parts=['train','val'],
    tasks=['bbox','class'],
    bbox_rotation='counter-clockwise',
    # project_folder='F:\\working',
    instance_names=[
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
        'other-airplane'])()
# print(fair1m_settings)

rarePlanes_settings = SettingsDataset(
    dataset_name='rarePlanes',
    dataset_parts=['val'],
    tasks=['bbox'],
    bbox_rotation='clockwise',
    filter_out_truncated=False,
    project_folder='/media/louis/Data/working',
    instance_names=[
       'Small Civil Transport/Utility',
       'Medium Civil Transport/Utility',
       'Large Civil Transport/Utility',
       'Military Transport/Utility/AWAC',
       'Military Bomber',
       'Military Fighter/Interceptor/Attack',
       'Military Trainer'])()
# instance_names=[
#         1, 2, 3])()

dataset_settings = [rarePlanes_settings]   # fair1m_settings]    # dota_settings]

save_cutouts=True
convert_to_imagenet=False
filter_images=False

if save_cutouts:
    for my_settings in dataset_settings:
        write_cutouts(my_settings, False)

if convert_to_imagenet:
    convert_my_labels_to_imagenet(rarePlanes_settings)

if filter_images:
    filter_truncated_images(dataset_settings[0])

# normalize_rarePlanes_annotations(dataset_settings[0])
