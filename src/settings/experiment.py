import os
import json

from src.settings.utils import create_folder, get_project_folder

# NOTES: The recognition patches are read from the json file (e.g.
# no_duplicates.json), while the detection patches are read from the
# folder (e.g. /data/Gaofen/train/images)

# TODO: Recognition patches paths for every dataset name


class Utils:
    """docstring for Utils"""

    def __init__(self, exp_name):
        self.exp_name = exp_name if exp_name is not None else 'exp_temp'
        self.project_folder = get_project_folder()
        self.experiment_folder = self.get_experiment_folder()

    def get_settings_path(self, update):
        # SETTINGS PATH
        settings_path = os.path.join(self.experiment_folder, "settings.json")
        print(f'The following settings will be used:\n{settings_path}\n')

        # READ EXISTING SETTINGS FILE
        if os.path.exists(settings_path) and not update:
            with open(settings_path, 'r') as f:
                return json.load(f)
        elif os.path.exists(settings_path) and update:
            ans = input(
                'A settings file already exists. Do you want to overwrite the settings file? [y/n] ')
            if ans != 'y':
                print('Please confirm it.')
                return 0
            else:
                print('\n')
        return settings_path

    def get_dataset_folder(self):
        dataset_folder = os.path.join(
            self.project_folder, "data", self.dataset_name)
        assert create_folder(dataset_folder)
        return dataset_folder

    def get_segmentation_folder(self, dataset_folder, dataset_part):
        segmentation_folder = os.path.join(
            dataset_folder, dataset_part, "segmentation")
        assert create_folder(segmentation_folder)
        return segmentation_folder
    def get_experiment_folder(self):
        # EXPERIMENTS FOLDER
        experiments_folder = os.path.join(self.project_folder, 'experiments')
        assert create_folder(experiments_folder)
        # EXPERIMENT FOLDER
        experiment_folder = os.path.join(experiments_folder, self.exp_name)
        assert create_folder(experiment_folder)
        return experiment_folder

    def get_train_log_path(self, model_name):
        train_log_path = os.path.join(
            self.experiment_folder,
            f"{model_name}_train.log")
        return train_log_path  # train_log_folder,

    def get_model_path(self, model_name, ext):
        model_path = os.path.join(
            self.experiment_folder,
            f"{model_name}.{ext}")
        return model_path


class SettingsSegmentation(Utils):
    """
        docstring for SettingsSegmentation
    """

    def __init__(self,
                 dataset_name,
                 save_patches=None,
                 patch_size=None,
                 exp_no=None,
                 exp_name=None,
                 patience=None,
                 epochs=None,
                 batch_size=None,
                 model_name=None,
                 init_features=None,
                 split_ratio=None,
                 learning_rate=None,
                 patch_config=None,
                 output_image=None,
                 bbox_rotation='clockwise',
                 update=True,
                 ):
        super(SettingsSegmentation, self).__init__(exp_name)

        # MODEL
        self.model_name = model_name
        self.init_features = init_features

        # EXPERIMENTS
        self.exp_name = exp_name
        self.exp_no = exp_no

        # TRAINING
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.output_image = output_image
        self.patch_config = patch_config

        # DATA
        self.dataset_name = dataset_name
        self.save_patches = save_patches
        self.patch_size = patch_size
        self.bbox_rotation = bbox_rotation

        # SETTINGS CONFIG
        self.update = update

    def __call__(self):
        return self.get_settings()

    def get_settings(self):
        settings_path = self.get_settings_path(self)
        print(settings_path)

        # DATASET DETAILS
        dataset_folder = self.get_dataset_folder()
        dataset_train_folders = self.get_dataset_part_folders(
            dataset_folder, 'train')
        dataset_val_folders = self.get_dataset_part_folders(
            dataset_folder, 'val')
        # dataset_train_folders = self.get_dataset_part_folders(dataset_folder,'train')

        # PATCH DETAILS
        train_patch_folders = self.get_patch_folders(dataset_folder, 'train')
        val_patch_folders = self.get_patch_folders(dataset_folder, 'val')

        settings = {
            'dataset': {
                'name': self.dataset_name,
                'bbox_rotation': self.bbox_rotation,
                'save_patches': self.save_patches,
                'train': {
                    'image_folder': dataset_train_folders[0],
                    'binary_mask_folder': dataset_train_folders[1],
                    'instance_mask_folder': dataset_train_folders[2],
                    'label_path': dataset_train_folders[3],
                    'bounding_box_folder': dataset_train_folders[4]
                },
                'val': {
                    'image_folder': dataset_val_folders[0],
                    'binary_mask_folder': dataset_val_folders[1],
                    'instance_mask_folder': dataset_val_folders[2],
                    'label_path': dataset_val_folders[3],
                    'bounding_box_folder': dataset_val_folders[4]
                },
            },
            'experiment': {
                'no': self.exp_no,
                'folder': self.experiment_folder,
                'name': self.exp_name
            },
            'model': {
                'name': self.model_name,
                'path': self.get_model_path(self.model_name, 'pth'),
                'init_features': self.init_features,
            },
            'patch': {
                'size': self.patch_size,
                'train': {
                    'img_folder': train_patch_folders[0],
                    'orthogonal_img_folder': train_patch_folders[1],
                    'orthogonal_zoomed_img_folder': train_patch_folders[2],
                    'mask_folder': train_patch_folders[3],
                    'orthogonal_mask_folder': train_patch_folders[4],
                    'orthogonal_zoomed_mask_folder': train_patch_folders[5],
                    'label_folder': train_patch_folders[6],
                },
                'val': {
                    'img_folder': val_patch_folders[0],
                    'orthogonal_img_folder': val_patch_folders[1],
                    'orthogonal_zoomed_img_folder': val_patch_folders[2],
                    'mask_folder': val_patch_folders[3],
                    'orthogonal_mask_folder': val_patch_folders[4],
                    'orthogonal_zoomed_mask_folder': val_patch_folders[5],
                    'label_folder': val_patch_folders[6],
                },
            },
            'training': {
                'patch_config': self.patch_config,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience,
                'split_ratio': self.split_ratio,
                'output_image': self.output_image
            }
        }
        with open(settings_path, 'w+') as f:
            json.dump(settings, f, indent=4)

        return settings

    def get_dataset_part_folders(self, dataset_folder, dataset_part):
        # ORIGINAL DATASET
        dataset_image_folder = os.path.join(
            dataset_folder, dataset_part, 'images')
        dataset_binary_mask_folder = os.path.join(
            dataset_folder, dataset_part, 'binary_masks')
        dataset_instance_mask_folder = os.path.join(
            dataset_folder, dataset_part, 'instance_masks')
        dataset_label_path = os.path.join(
            dataset_folder,
            dataset_part,
            'labels',
            f'iSAID_{dataset_part}.json')
        dataset_bbox_folder = os.path.join(
            dataset_folder, dataset_part, 'bounding_boxes')
        return dataset_image_folder, dataset_binary_mask_folder, dataset_instance_mask_folder, dataset_label_path, dataset_bbox_folder

    def get_patch_folders(self, dataset_folder, dataset_part):
        patch_folder_base = os.path.join(
            self.get_segmentation_folder(
                dataset_folder,
                dataset_part),
            f"patches_{self.patch_size}")
        img_patch_folder = os.path.join(patch_folder_base, 'images')
        orthogonal_img_patch_folder = os.path.join(
            patch_folder_base, 'orthogonal_images')
        orthogonal_zoomed_img_patch_folder = os.path.join(
            patch_folder_base, 'orthogonal_zoomed_images')
        mask_patch_folder = os.path.join(patch_folder_base, 'masks')
        orthogonal_mask_patch_folder = os.path.join(
            patch_folder_base, 'orthogonal_masks')
        orthogonal_zoomed_mask_patch_folder = os.path.join(
            patch_folder_base, 'orthogonal_zoomed_masks')
        label_folder = os.path.join(patch_folder_base, 'labels')

        folders = [patch_folder_base,
                   img_patch_folder,
                   orthogonal_img_patch_folder,
                   orthogonal_zoomed_img_patch_folder,
                   mask_patch_folder,
                   orthogonal_mask_patch_folder,
                   orthogonal_zoomed_mask_patch_folder,
                   label_folder]

        for folder in folders:
            assert create_folder(folder)
        return folders[1:]


class SettingsRecognition(Utils):
    def __init__(self,
                 model_name=None,
                 exp_name=None,
                 # dataset_name=None,
                 settings_datasets=[],
                 patch_size=None,
                 batch_size=None,
                 epochs=None,
                 hot_encoding=None,
                 split_ratio=None,
                 patch_config=None,
                 class_weight=None,
                 update=True
                 ):
        from src.main_segmentation import exp_no
        super(SettingsRecognition, self).__init__(exp_no)
        self.update = update
        self.model_name = model_name
        # DATASETS
        self.settings_datasets = settings_datasets
        # EXPERIMENT
        self.exp_name = exp_name
        self.patch_size = patch_size

        # TRAINING HYPERPARAMETERS
        self.hot_encoding = hot_encoding
        self.batch_size = batch_size
        self.epochs = epochs
        self.split_ratio = split_ratio
        self.patch_config = patch_config
        self.class_weight = class_weight

    def __call__(self):
        return self.get_settings()

    def get_settings(self):
        from src.main_detection import dataset_name
        # SETTINGS PATH
        settings_path = self.get_settings_path(self.update)

        settings = {
            # 'project_folder':self.project_folder,
            'experiment': {
                # 'no':self.exp_no,
                'folder': self.experiment_folder,
                'name': self.exp_name
            },
            'model': {
                'name': self.model_name,
                'path': self.get_model_path(self.model_name, 'pth'),
            },

            'dataset': {
                'name': dataset_name,
                'folder': self.get_dataset_folder(dataset_name),
                'instance_table': self.get_instance_table(dataset_name)
            },

            'training': {
                'class_weight': self.class_weight,
                'patch_config': self.patch_config,
                'hot_encoding': self.hot_encoding,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'log_path': self.get_train_log_path(self.model_name),
                # split into train, test, val after merging datasets if None,
                # no split (stick to the original folders)
                'split_ratio': self.split_ratio,
            },
            # 'testing': {
            #     'fig_folder':self.get_test_fig_folder()
            # }
        }

        with open(settings_path, 'w+') as f:
            json.dump(settings, f, indent=4)

        return settings

    def get_instance_table(self, dataset_name):
        instance_table = {}
        if dataset_name == 'Gaofen':
            instance_names = [
                'other',
                'ARJ21',
                'Boeing737',
                'Boeing747',
                'Boeing777',
                'Boeing787',
                'A220',
                'A321',
                'A330',
                'A350']
            instance_table = {
                instance_name: i for i,
                instance_name in enumerate(instance_names)}
        return instance_table

    def get_patch_folders(self, dataset_folder, dataset_part):
        # PATCH SAVE DIRECTORIES
        # EX:
        # /home/murat/Projects/airplane_recognition/data/Gaofen/train/recognition/patches_128
        patch_folder_base = os.path.join(
            self.get_recognition_folder(
                dataset_folder,
                dataset_part),
            f"patches_{self.patch_size}")
        img_patch_folder = f"{patch_folder_base}/images"
        img_patch_orthogonal_folder = f"{patch_folder_base}/orthogonal_images"
        img_patch_orthogonal_zoomed_folder = f"{patch_folder_base}/orthogonal_zoomed_images"
        label_patch_folder = f"{patch_folder_base}/labels"

        folders = [patch_folder_base,
                   img_patch_folder,
                   img_patch_orthogonal_folder,
                   img_patch_orthogonal_zoomed_folder,
                   label_patch_folder]

        # IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders:
            assert create_folder(folder)
        return folders


class SettingsDetection(Utils):
    """docstring for SettingsDetection"""

    def __init__(self,
                 update=True,
                 model_name=None,
                 exp_name=None,
                 patch_size=None,
                 overlap=100,
                 split_ratio=None,
                 batch_size=None,
                 patience=None,
                 save_patches=None,
                 dataset_name=None,
                 box_corner_threshold=2,
                 label_names=None,
                 epochs=None,
                 ):
        super(SettingsDetection, self).__init__(exp_name)
        self.update = update

        # TRAINING CONFIGS
        self.model_name = model_name
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.split_ratio = split_ratio

        # DATASET
        self.label_names = label_names
        self.save_patches = save_patches
        self.dataset_name = dataset_name
        # PATCH CONFIGS
        self.patch_size = patch_size
        self.overlap = overlap
        self.box_corner_threshold = box_corner_threshold

    def __call__(self):
        return self.get_settings()

    def get_settings(self):
        # SETTINGS PATH
        settings_path = self.get_settings_path(self.update)

        # DATASET DETAILS
        dataset_folder = self.get_dataset_folder(self.dataset_name)

        # PATCH FOLDERS
        patch_train_folders = self.get_patch_folders(dataset_folder, 'train')
        patch_test_folders = self.get_patch_folders(dataset_folder, 'test')
        patch_val_folders = self.get_patch_folders(dataset_folder, 'val')

        # ORIGINAL DATA FOLDERS
        train_folders = self.get_original_data_folders(dataset_folder, 'train')
        test_folders = self.get_original_data_folders(dataset_folder, 'test')
        val_folders = self.get_original_data_folders(dataset_folder, 'val')

        settings = {
            'project_folder': self.project_folder,
            'experiment': {
                'folder': self.experiment_folder,
                'name': self.exp_name
            },
            'model': {
                'name': self.model_name,
                'path': self.get_model_path(self.model_name, 'pth'),
            },

            'dataset': {
                'label_names': self.label_names,
                'save_patches': self.save_patches,
                'folder': dataset_folder,
                'name': self.dataset_name,
                'train': {'image_folder': train_folders[0],
                          'label_folder': train_folders[1],
                          },
                'test': {'image_folder': test_folders[0],
                         'label_folder': test_folders[1],
                         },
                'val': {'image_folder': val_folders[0],
                        'label_folder': val_folders[1],
                        },

            },
            'patch': {
                'overlap': self.overlap,
                'box_corner_threshold': self.box_corner_threshold,
                'size': self.patch_size,
                'train': {
                    'folder_base': patch_train_folders[0],
                    'image_folder': patch_train_folders[1],
                    'label_folder': patch_train_folders[2],
                    'label_binary_yolo_folder': patch_train_folders[3],
                    'label_dota_folder': patch_train_folders[4],
                    'label_binary_dota_folder': patch_train_folders[5],
                },
                'test': {
                    'folder_base': patch_test_folders[0],
                    'image_folder': patch_test_folders[1],
                    'label_folder': patch_test_folders[2],
                    'label_binary_yolo_folder': patch_test_folders[3],
                    'label_dota_folder': patch_test_folders[4],
                    'label_binary_dota_folder': patch_test_folders[5],
                },
                'val': {
                    'folder_base': patch_val_folders[0],
                    'image_folder': patch_val_folders[1],
                    'label_folder': patch_val_folders[2],
                    'label_binary_yolo_folder': patch_val_folders[3],
                    'label_dota_folder': patch_val_folders[4],
                    'label_binary_dota_folder': patch_val_folders[5],
                },
            },
            'training': {
                # 'resume':self.resume,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'log_path': self.get_train_log_path(self.model_name),
                # split into train, test, val after merging datasets if None,
                # no split (stick to the original folders)
                'split_ratio': self.split_ratio,
            },

        }

        # try:
        if self.model_name.startswith('yolo'):
            # if True: # If YOLO
            settings['training']['yolo'] = {'train_py': os.path.join(self.project_folder, 'yolo_models/yolov5/train.py'),
                                            'data_yaml': os.path.join(self.experiment_folder, 'data.yaml'),
                                            'hyp_config_yaml': os.path.join(self.experiment_folder, 'hyp_config.yaml'),
                                            'weights_yaml': os.path.join(self.project_folder, f'yolo_models/yolov5/models/{self.model_name}.yaml'),
                                            'weights': os.path.join(self.experiment_folder, 'exp', 'weights', 'best.pt')
                                            }
            settings['testing'] = {
                'yolo':
                {
                    'test_py': os.path.join(self.project_folder, 'yolo_models/yolov5/val.py'),
                }
            }

            settings = self.get_yolo_symlink_folders(settings)

        # except Exception:
            # print('Not a YOLO model.\n')

        with open(settings_path, 'w+') as f:
            json.dump(settings, f, indent=4)
        return settings

    def get_detection_folder(self, dataset_folder, dataset_part):
        detection_folder = os.path.join(
            dataset_folder, dataset_part, "detection")
        folder_ok = create_folder(detection_folder)
        if not folder_ok:
            return 0
        return detection_folder

    def get_original_data_folders(self, dataset_folder, dataset_part):
        original_folder_base = os.path.join(dataset_folder, dataset_part)

        img_folder = os.path.join(original_folder_base, 'images')
        label_folder = os.path.join(original_folder_base, 'labels')
        # label_dota_folder = os.path.join(original_folder_base,'labels_dota')
        # label_binary_dota_folder = os.path.join(original_folder_base,'labels_binary_dota')

        folders = [original_folder_base,
                   img_folder,
                   label_folder]

        # IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders:
            assert create_folder(folder)
        return img_folder, label_folder

    def get_patch_folders(self, dataset_folder, dataset_part):
        patch_folder_base = os.path.join(
            self.get_detection_folder(
                dataset_folder,
                dataset_part),
            f"patches_{self.patch_size}")
        # patch_folder = f"{self.data_folder}/{self.dataset_name}/patches_{patch_size}"
        img_patch_folder = os.path.join(patch_folder_base, "images")
        label_patch_folder = os.path.join(patch_folder_base, "labels")
        label_binary_patch_yolo_folder = os.path.join(
            patch_folder_base, "labels_binary_yolo")
        label_patch_dota_folder = os.path.join(
            patch_folder_base, "labels_dota")
        label_binary_patch_dota_folder = os.path.join(
            patch_folder_base, "labels_binary_dota")

        folders = [patch_folder_base,
                   img_patch_folder,
                   label_patch_folder,
                   label_binary_patch_yolo_folder,
                   label_patch_dota_folder,
                   label_binary_patch_dota_folder]

        # IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders:
            assert create_folder(folder)
        return folders

    def get_yolo_symlink_folders(self, settings):

        symlink_root_folder = os.path.join(
            self.experiment_folder, 'data_symlinks')
        train_symlink_root_folder = os.path.join(symlink_root_folder, 'train')
        test_symlink_root_folder = os.path.join(symlink_root_folder, 'test')
        val_symlink_root_folder = os.path.join(symlink_root_folder, 'val')

        train_img_symlink_folder = os.path.join(
            train_symlink_root_folder, 'images')
        test_img_symlink_folder = os.path.join(
            test_symlink_root_folder, 'images')
        val_img_symlink_folder = os.path.join(
            val_symlink_root_folder, 'images')

        train_label_symlink_folder = os.path.join(
            train_symlink_root_folder, 'labels')
        test_label_symlink_folder = os.path.join(
            test_symlink_root_folder, 'labels')
        val_label_symlink_folder = os.path.join(
            val_symlink_root_folder, 'labels')

        folders = [
            symlink_root_folder,
            train_symlink_root_folder,
            test_symlink_root_folder,
            val_symlink_root_folder,
            train_img_symlink_folder,
            test_img_symlink_folder,
            val_img_symlink_folder,
            train_label_symlink_folder,
            test_label_symlink_folder,
            val_label_symlink_folder
        ]
        for folder in folders:
            assert create_folder(folder)

        settings['patch']['train']['image_folder_if_split'] = train_img_symlink_folder
        settings['patch']['train']['label_binary_yolo_folder_if_split'] = train_label_symlink_folder
        settings['patch']['test']['image_folder_if_split'] = test_img_symlink_folder
        settings['patch']['test']['label_binary_yolo_folder_if_split'] = test_label_symlink_folder
        settings['patch']['val']['image_folder_if_split'] = val_img_symlink_folder
        settings['patch']['val']['label_binary_yolo_folder_if_split'] = val_label_symlink_folder
        return settings


if __name__ == '__main__':
    # settings_detection = SettingsDetection(patch_size=512)()
    # print(settings_detection)

    # settings_recognition = SettingsRecognition(dataset_names=['Gaofen','DOTA'])()
    # print(settings_recognition['datasets'][1])

    settings_segmentation = SettingsSegmentation(dataset_name='DOTA',
                                                 patch_size=128)()
    print(settings_segmentation)
    # print('mask_folder' in settings_segmentation['patch']['train'].keys())
