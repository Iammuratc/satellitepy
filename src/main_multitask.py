import os.path

from settings.dataset import SettingsDataset
from utilities import write_cutouts

### DATASETS
save_cutouts=True

dota_settings = SettingsDataset(
    dataset_name='DOTA',
    dataset_parts=['train','val'],
    tasks=['bbox','seg'],
    bbox_rotation='clockwise',
    instance_names=['plane'])()
# print(dota_settings)

fair1m_settings = SettingsDataset(
    dataset_name='fair1m',
    dataset_parts=['train','val'],
    tasks=['bbox','class'],
    bbox_rotation='counter-clockwise',
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
    dataset_parts=['train','val'],
    tasks=['bbox'],
    bbox_rotation='clockwise',
    instance_names=[
        'Small Civil Transport/Utility',
        'Medium Civil Transport/Utility',
        'Large Civil Transport/Utility',
        'Military Transport/Utility/AWAC',
        'Military Bomber',
        'Military Fighter/Interceptor/Attack',
        'Military Trainer'])()

dataset_settings = [rarePlanes_settings] #fair1m_settings]  # dota_settings]

if save_cutouts:
    for my_settings in dataset_settings:
        write_cutouts(my_settings, False)
