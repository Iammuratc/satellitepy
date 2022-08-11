import os
import json


#NOTES: The recognition patches are read from the json file (e.g. no_duplicates.json), while the detection patches are read from the folder (e.g. /data/Gaofen/train/images)

class SettingsUtils:
    """docstring for SettingsUtils"""
    def __init__(self,exp_no):
        self.exp_no = exp_no if exp_no != None else 'temp'
        self.project_folder = self.get_project_folder()
        self.experiment_folder = self.get_experiment_folder()

    def get_project_folder(self):
        project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert self.create_folder(project_folder)
        return project_folder

    def get_dataset_folder(self,dataset_name):
        dataset_folder = os.path.join(self.project_folder,"data",dataset_name)
        assert self.create_folder(dataset_folder)
        return dataset_folder


    def get_experiment_folder(self):
        ### EXPERIMENTS FOLDER
        experiments_folder=os.path.join(self.project_folder,'experiments')
        folder_ok = self.create_folder(experiments_folder)
        if not folder_ok:
            return 0
        ### EXPERIMENT FOLDER
        experiment_folder=os.path.join(experiments_folder,f'exp_{self.exp_no}')
        folder_ok = self.create_folder(experiment_folder)
        if not folder_ok:
            return 0
        return experiment_folder

    def get_json_file_path(self,dataset_part):
        return os.path.join(self.get_recognition_folder(dataset_part),"no_duplicates.json")

    def create_folder(self,folder):
        if not os.path.exists(folder):
            msg = f'The following folder will be created:\n{folder}\nDo you confirm?[y/n] '
            ans = input(msg)
            if ans == 'y':
                os.makedirs(folder,exist_ok=True)
                return 1
            print('Please confirm it.')
            return 0
        else:
            return 1


    def get_train_log_path(self,model_name):
        train_log_path = os.path.join(self.experiment_folder,f"{model_name}_train.log")
        return train_log_path #train_log_folder,


class Settings(SettingsUtils):
    def __init__(self,  model_name=None,
                        exp_no=None,
                        exp_name=None,
                        patch_size=None,
                        batch_size=None,
                        epochs=None,
                        hot_encoding=None,
                        split_ratio=None,
                        patch_config=None,
                        class_weight=None,
                        update=True
                        ):
        super(Settings, self).__init__(exp_no)
        self.update=update
        self.model_name = model_name
        
        self.exp_name = exp_name
        self.patch_size = patch_size

        ### TRAINING HYPERPARAMETERS
        self.hot_encoding=hot_encoding
        self.batch_size=batch_size
        self.epochs=epochs
        self.split_ratio=split_ratio
        self.patch_config=patch_config
        self.class_weight=class_weight


    def __call__(self):
        return self.get_settings()

    def get_settings(self):
        ### SETTINGS PATH
        settings_path = os.path.join(self.experiment_folder,"settings.json")

        print(f'The following settings will be used:\n{settings_path}\n')

        ## READ EXISTING SETTINGS FILE
        if os.path.exists(settings_path) and not self.update:
            with open(settings_path,'r') as f:
                return json.load(f)

        ### DATASET DETAILS
        dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
        dataset_name = 'Gaofen'
        self.dataset_folder = self.get_dataset_folder(dataset_name)

        ### PATCH FOLDERS
        train_folders = self.get_patch_folders('train')
        test_folders = self.get_patch_folders('test')
        val_folders = self.get_patch_folders('val')


        settings = {
            'project_folder':self.project_folder,
            'experiment': {
                'no':self.exp_no,
                'folder':self.experiment_folder,
                'name':self.exp_name            
            },
            'model': {
                'name':self.model_name,
                'path':self.get_model_path(),
            },

            'dataset': {
                'folder':self.dataset_folder,
                'name':dataset_name,
                'id': dataset_id,
                'instance_table':self.get_instance_table()
            },
            'patch': {
                'size': self.patch_size,
                'train': {
                    'json_file_path':self.get_json_file_path('train'),
                    'patch_folder':train_folders[0],
                    'img_patch_folder':train_folders[1],
                    'img_patch_orthogonal_folder':train_folders[2],
                    'img_patch_orthogonal_zoomed_folder':train_folders[3],
                    'label_patch_folder':train_folders[4],
                },
                'test': {
                    'json_file_path':self.get_json_file_path('test'),
                    'patch_folder':test_folders[0],
                    'img_patch_folder':test_folders[1],
                    'img_patch_orthogonal_folder':test_folders[2],
                    'img_patch_orthogonal_zoomed_folder':test_folders[3],
                    'label_patch_folder':test_folders[4],
                },
                'val': {
                    'json_file_path':self.get_json_file_path('val'),
                    'patch_folder':val_folders[0],
                    'img_patch_folder':val_folders[1],
                    'img_patch_orthogonal_folder':val_folders[2],
                    'img_patch_orthogonal_zoomed_folder':val_folders[3],
                    'label_patch_folder':val_folders[4],
                }
            },
            'training': {
                'class_weight':self.class_weight,
                'patch_config':self.patch_config,
                'hot_encoding':self.hot_encoding,
                'batch_size':self.batch_size,
                'epochs':self.epochs,
                'log_path':self.get_train_log_path(self.model_name),
                'split_ratio':self.split_ratio, # split into train, test, val after merging datasets if None, no split (stick to the original folders)
            },
            # 'testing': {
            #     'fig_folder':self.get_test_fig_folder()
            # }
        }

        with open(settings_path,'w+') as f:
            json.dump(settings,f,indent=4)
        
        return settings

    def get_recognition_folder(self,dataset_part):
        recognition_folder = os.path.join(self.dataset_folder,dataset_part,"recognition")
        folder_ok = self.create_folder(recognition_folder)
        if not folder_ok:
            return 0        
        return recognition_folder


    def get_instance_table(self):
        instance_names = ['other', 'ARJ21','Boeing737', 'Boeing747','Boeing777', 'Boeing787', 'A220', 'A321', 'A330', 'A350']
        instance_table = { instance_name:i for i,instance_name in enumerate(instance_names)}
        return instance_table


    def get_model_path(self):
        ### ASK USER TO CREATE THE FOLDER
        model_path = os.path.join(self.experiment_folder,f"{self.model_name}.pth")

        return model_path


    def get_patch_folders(self,dataset_part):
        ### PATCH SAVE DIRECTORIES
        ### EX: /home/murat/Projects/airplane_recognition/data/Gaofen/train/recognition/patches_128
        patch_folder_base = os.path.join(self.get_recognition_folder(dataset_part),f"patches_{self.patch_size}")
        img_patch_folder = f"{patch_folder_base}/images"
        img_patch_orthogonal_folder = f"{patch_folder_base}/orthogonal_images"
        img_patch_orthogonal_zoomed_folder = f"{patch_folder_base}/orthogonal_zoomed_images"
        label_patch_folder = f"{patch_folder_base}/labels"

        folders = [ patch_folder_base,
                    img_patch_folder,
                    img_patch_orthogonal_folder,
                    img_patch_orthogonal_zoomed_folder,
                    label_patch_folder]


        ## IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders:
            folder_ok = self.create_folder(folder)
            if not folder_ok:
                return 0
        return folders


class SettingsDetection(SettingsUtils):
    """docstring for SettingsDetection"""
    def __init__(self, 
                update=True,
                model_name=None,
                exp_no=None,
                exp_name=None,
                patch_size=None,
                overlap=100,
                split_ratio=None,
                batch_size=None,
                patience=None,
                # resume=None,
                box_corner_threshold=2,
                epochs=None,
                        ):
        super(SettingsDetection, self).__init__(exp_no)
        self.update=update

        ### TRAINING CONFIGS
        self.model_name = model_name
        self.exp_name = exp_name
        self.batch_size=batch_size
        self.epochs=epochs
        self.patience=patience
        # self.resume=resume

        ### PATCH CONFIGS 
        self.patch_size = patch_size
        self.split_ratio=split_ratio
        self.overlap=overlap
        self.box_corner_threshold=box_corner_threshold

    def __call__(self):
        return self.get_settings()

    def get_settings(self):
        ### SETTINGS PATH
        settings_path = os.path.join(self.experiment_folder,"settings.json")

        print(f'The following settings will be used:\n{settings_path}\n')


        ## READ EXISTING SETTINGS FILE
        if os.path.exists(settings_path) and not self.update:
            with open(settings_path,'r') as f:
                return json.load(f)

        ### DATASET DETAILS
        dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
        dataset_name = 'Gaofen'
        self.dataset_folder = self.get_dataset_folder(dataset_name)


        ### PATCH FOLDERS
        patch_train_folders = self.get_patch_folders('train')
        patch_test_folders = self.get_patch_folders('test')
        patch_val_folders = self.get_patch_folders('val')


        ### ORIGINAL DATA FOLDERS
        train_folders = self.get_original_data_folders('train')
        test_folders = self.get_original_data_folders('test')
        val_folders = self.get_original_data_folders('val')

        settings = {
            'project_folder':self.project_folder,
            'experiment': {
                'no':self.exp_no,
                'folder':self.experiment_folder,
                'name':self.exp_name            
            },
            'model': {
                'name':self.model_name,
                'path':self.get_model_path(),
            },

            'dataset': {
                'folder':self.dataset_folder,
                'name':dataset_name,
                'id': dataset_id,
                'train': {  'image_folder':train_folders[0],
                            'label_folder':train_folders[1],
                        },
                'test': {   'image_folder':test_folders[0],
                            'label_folder':test_folders[1],
                        },
                'val': {    'image_folder':val_folders[0],
                            'label_folder':val_folders[1],
                        },

                },
            'patch' : {
                'overlap':self.overlap,
                'box_corner_threshold':self.box_corner_threshold,
                'size':self.patch_size,
                'train': {
                    'patch_folder_base':patch_train_folders[0],
                    'img_patch_folder':patch_train_folders[1],
                    'label_patch_folder':patch_train_folders[2],
                    'label_patch_yolo_folder':patch_train_folders[3],
                },
                'test': {
                    'patch_folder_base':patch_test_folders[0],
                    'img_patch_folder':patch_test_folders[1],
                    'label_patch_folder':patch_test_folders[2],
                    'label_patch_yolo_folder':patch_test_folders[3],
                },
                'val': {
                    'patch_folder_base':patch_val_folders[0],
                    'img_patch_folder':patch_val_folders[1],
                    'label_patch_folder':patch_val_folders[2],
                    'label_patch_yolo_folder':patch_val_folders[3],
                },
            },
            'training': {
                'yolo': {   'train_py':os.path.join(self.project_folder,'yolo_models/yolov5/train.py'),
                            'data_yaml':os.path.join(self.experiment_folder,'data.yaml'),
                            'hyp_config_yaml':os.path.join(self.experiment_folder,'hyp_config.yaml'),
                            'weights_yaml':os.path.join(self.project_folder,f'yolo_models/yolov5/models/{self.model_name}.yaml'),
                            'weights':os.path.join(self.experiment_folder,'exp','weights','best.pt')

                },
                # 'resume':self.resume,
                'patience':self.patience,
                'batch_size':self.batch_size,
                'epochs':self.epochs,
                'log_path':self.get_train_log_path(self.model_name),
                'split_ratio':self.split_ratio, # split into train, test, val after merging datasets if None, no split (stick to the original folders)
            },

            'testing': 
            {
                'yolo': 
                {
                    'test_py':os.path.join(self.project_folder,'yolo_models/yolov5/val.py'),
                }
            }
        }
        
        with open(settings_path,'w+') as f:
            json.dump(settings,f,indent=4)
        return settings

    def get_detection_folder(self,dataset_part):
        detection_folder = os.path.join(self.dataset_folder,dataset_part,"detection")
        folder_ok = self.create_folder(detection_folder)
        if not folder_ok:
            return 0        
        return detection_folder

    def get_original_data_folders(self,dataset_part):
        original_folder_base = os.path.join(self.dataset_folder,dataset_part)

        img_folder = os.path.join(original_folder_base,'images')
        label_folder = os.path.join(original_folder_base,'label_xml')
         
        folders = [original_folder_base,
                    img_folder,
                    label_folder]
        
        ## IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders:
            folder_ok = self.create_folder(folder)
            if not folder_ok:
                return 0
        return img_folder, label_folder

    def get_patch_folders(self,dataset_part):
        patch_folder_base = os.path.join(self.get_detection_folder(dataset_part),f"patches_{self.patch_size}")
        # patch_folder = f"{self.data_folder}/{self.dataset_name}/patches_{patch_size}"
        img_patch_folder = os.path.join(patch_folder_base,"images")
        label_patch_folder = os.path.join(patch_folder_base,"labels")
        label_patch_yolo_folder = os.path.join(patch_folder_base,"labels_yolo")

        folders = [ patch_folder_base,
                    img_patch_folder,
                    label_patch_folder,
                    label_patch_yolo_folder]

        ## IF THEY DO NOT EXIST, CREATE THEN
        for folder in folders:
            folder_ok = self.create_folder(folder)
            if not folder_ok:
                return 0
        return folders


    def get_model_path(self):
        ### ASK USER TO CREATE THE FOLDER
        model_path = os.path.join(self.experiment_folder,f"{self.model_name}.pt")
        return model_path

if __name__ == '__main__':
    settings_detection = SettingsDetection(patch_size=512)()
    print(settings_detection)