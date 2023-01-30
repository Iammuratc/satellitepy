import os
from .utils import create_folder, get_project_folder#, get_logger


#TODO:
#   Log files
class Utils:
    """docstring for Utils"""

    def __init__(self, dataset_name, tasks, project_folder):
        super(Utils, self).__init__()
        self.dataset_name = dataset_name
        self.tasks = tasks
        self.project_folder = get_project_folder(project_folder)
        self.dataset_folder = self.get_dataset_folder()

    def get_dataset_folder(self):
        dataset_folder = os.path.join(
            self.project_folder, "data", self.dataset_name)
        assert create_folder(dataset_folder)
        return dataset_folder

    def get_cutout_folder(self, dataset_part):
        cutout_folder = os.path.join(
            self.dataset_folder, dataset_part, "cutouts")
        assert create_folder(cutout_folder)
        return cutout_folder

    def get_cutout_folders(self, dataset_part):
        # PATCH SAVE DIRECTORIES
        cutout_folder_root = self.get_cutout_folder(dataset_part)
        img_cutout_folder = os.path.join(cutout_folder_root, "images")
        img_cutout_orthogonal_folder = os.path.join(
            cutout_folder_root, "orthogonal_images")
        img_cutout_orthogonal_zoomed_folder = os.path.join(
            cutout_folder_root, "orthogonal_zoomed_images")
        label_cutout_folder = os.path.join(cutout_folder_root, "labels")

        folders = {
            'root_folder':cutout_folder_root,
            'image_folder': img_cutout_folder,
            'orthogonal_image_folder': img_cutout_orthogonal_folder,
            'orthogonal_zoomed_image_folder': img_cutout_orthogonal_zoomed_folder,
            'label_folder': label_cutout_folder
        }
        # if 'class' in self.tasks:
        #     imagenet_label_file = os.path.join(cutout_folder_root,'imagenet_labels.txt')
        #     folders['imagenet_label_file']=imagenet_label_file

        if 'filter' in self.tasks:
            full_image_folder = os.path.join(cutout_folder_root, "full_images")
            full_orthogonal_image_folder = os.path.join(cutout_folder_root, "full_orthogonal_images")
            full_orthogonal_zoomed_image_folder = os.path.join(cutout_folder_root, "full_orthogonal_zoomed_images")
            folders['full_image_folder'] = full_image_folder
            folders['full_orthogonal_image_folder'] = full_orthogonal_image_folder
            folders['full_orthogonal_zoomed_image_folder'] = full_orthogonal_zoomed_image_folder

        if 'seg' in self.tasks:
            seg_folders = {
                'mask_folder': os.path.join(cutout_folder_root, 'masks'),
                'orthogonal_mask_folder': os.path.join(cutout_folder_root, 'orthogonal_masks'),
                'orthogonal_zoomed_mask_folder': os.path.join(cutout_folder_root, 'orthogonal_zoomed_masks'),
            }
            folders.update(seg_folders)
        # IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders.values():
            assert create_folder(folder)
        return folders

    def get_original_data_folders(self, dataset_part):
        '''
        Get the original data folders, so the cutouts can be created from the original data
        '''
        original_folder_base = os.path.join(self.dataset_folder, dataset_part)

        img_folder = os.path.join(original_folder_base, 'images')
        # label_folder = os.path.join(original_folder_base, 'labels')

        folders = {
            'base_folder': original_folder_base,
            'image_folder': img_folder,
            'bounding_box_folder': os.path.join(self.dataset_folder, dataset_part, 'bounding_boxes')
            # 'label_folder': label_folder
        }
        if 'seg' in self.tasks:
            seg_folders = {
                'instance_mask_folder': os.path.join(self.dataset_folder, dataset_part, 'instance_masks'),
            }
            folders.update(seg_folders)
            # dataset_binary_mask_folder = os.path.join(dataset_folder,dataset_part,'binary_masks')

        # IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders.values():
            assert create_folder(folder)
        return folders

    # def init_logger(self,dataset_part,file,name):


class SettingsDataset(Utils):
    """docstring for SettingsDataset"""

    def __init__(self,
                 dataset_name,
                 tasks,
                 dataset_parts,
                 instance_names,
                 bbox_rotation,
                 project_folder=''):
        super(SettingsDataset, self).__init__(
            dataset_name=dataset_name,
            tasks=tasks,
            project_folder=project_folder)
        self.dataset_parts = dataset_parts
        self.instance_names = instance_names
        self.project_folder = get_project_folder(project_folder)
        self.bbox_rotation=bbox_rotation
    def __call__(self):
        return self.get_settings()

    def get_settings(self):

        # log_names = f'{self.dataset_name}_log'
        # log_file = os.path.join(self.get_cutout_folder(),'cutouts.log')
        # logger = get_logger(log_name,log_file)



        settings = {
            # 'log_names':log_name,
            'dataset_name':self.dataset_name,
            'tasks': self.tasks,
            'dataset_parts':self.dataset_parts,
            'instance_names':{instance_name:i for i,instance_name in enumerate(self.instance_names)},
            'bbox_rotation':self.bbox_rotation,
            'original': {
                dataset_part: self.get_original_data_folders(dataset_part)
                for dataset_part in self.dataset_parts
            },
            'cutout': {
                dataset_part: self.get_cutout_folders(dataset_part)
                for dataset_part in self.dataset_parts
            }
        }
        return settings


if __name__ == '__main__':
    dota_settings = SettingsDataset(
        dataset_name='DOTA',
        dataset_parts=['train', 'val'],
        tasks=['bbox', 'seg'],
        instance_names=['airplane'])()
    print(dota_settings)
