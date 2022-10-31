from settings.dataset import SettingsDataset
from utilities import write_cutouts, convert_my_labels_to_imagenet


save_cutouts=False

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

dataset_settings = [fair1m_settings]#,dota_settings]

if save_cutouts:
    for my_settings in dataset_settings:
        write_cutouts(my_settings,multi_process=True)

convert_my_labels_to_imagenet(fair1m_settings)