import os
import json


## TODO: move all the training paths in Recognition to settings

class Settings(object):
    """docstring for Settings"""
    def __init__(self,  model_name=None,
                        exp_no=None,
                        exp_name=None,
                        patch_size=None,
                        batch_size=None,
                        epochs=None,
                        hot_encoding=None,
                        merge_and_split_data=None,
                        split_ratio=None,
                        patch_config=None,
                        class_weight=None,
                        update=True):
        super(Settings, self).__init__()
        self.update=update
        self.model_name = model_name
        self.exp_no = exp_no
        self.exp_name = exp_name
        self.patch_size = patch_size

        ### TRAINING HYPERPARAMETERS
        self.hot_encoding=hot_encoding
        self.batch_size=batch_size
        self.epochs=epochs
        self.merge_and_split_data=merge_and_split_data
        self.split_ratio=split_ratio
        self.patch_config=patch_config
        self.class_weight=class_weight

        self.project_folder = self.get_project_folder()
        self.experiment_folder = self.get_experiment_folder()

    def __call__(self):
        return self.get_settings()

    def get_settings(self):
        ### SETTINGS PATH
        settings_path = os.path.join(self.experiment_folder,f"settings.json")

        print(f'The following settings will be used:\n{settings_path}\n')

        ## READ EXISTING SETTINGS FILE
        if os.path.exists(settings_path) and not self.update:
            with open(settings_path,'r') as f:
                return json.load(f)

        ### DATASET DETAILS
        dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
        dataset_name = 'Gaofen'

        ### PATCH FOLDERS
        dataset_folder = f"{self.project_folder}/data/{dataset_name}"
        train_folders = self.get_patch_folders(dataset_folder,'train')[2:]
        test_folders = self.get_patch_folders(dataset_folder,'test')[2:]
        val_folders = self.get_patch_folders(dataset_folder,'val')[2:]


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
                'folder':dataset_folder,
                'name':dataset_name,
                'id': dataset_id,
                'instance_table':self.get_instance_table()
            },
            'patch': {
                'size': self.patch_size,
                'train': {
                    'img_patch_folder':train_folders[0],
                    'img_patch_orthogonal_folder':train_folders[1],
                    'img_patch_orthogonal_zoomed_folder':train_folders[2],
                    'label_patch_folder':train_folders[3],
                },
                'test': {
                    'img_patch_folder':test_folders[0],
                    'img_patch_orthogonal_folder':test_folders[1],
                    'img_patch_orthogonal_zoomed_folder':test_folders[2],
                    'label_patch_folder':test_folders[3],
                },
                'val': {
                    'img_patch_folder':val_folders[0],
                    'img_patch_orthogonal_folder':val_folders[1],
                    'img_patch_orthogonal_zoomed_folder':val_folders[2],
                    'label_patch_folder':val_folders[3],
                }
            },
            'training': {
                'class_weight':self.class_weight,
                'patch_config':self.patch_config,
                'hot_encoding':self.hot_encoding,
                'batch_size':self.batch_size,
                'epochs':self.epochs,
                'log_path':self.get_train_log_path(),
                'merge_and_split_data':self.merge_and_split_data,
                'split_ratio':self.split_ratio, # split into train, test, val after merging datasets
            },
            # 'testing': {
            #     'fig_folder':self.get_test_fig_folder()
            # }
        }

        with open(settings_path,'w+') as f:
            json.dump(settings,f,indent=4)
        
        return settings

    def get_project_folder(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    def get_instance_table(self):
        instance_names = ['other', 'ARJ21','Boeing737', 'Boeing747','Boeing777', 'Boeing787', 'A220', 'A321', 'A330', 'A350']
        instance_table = { instance_name:i for i,instance_name in enumerate(instance_names)}
        return instance_table


    def get_model_path(self):
        ### ASK USER TO CREATE THE FOLDER
        model_path = os.path.join(self.experiment_folder,f"{self.model_name}.pth")

        return model_path

    def get_train_log_path(self):
        train_log_path = os.path.join(self.experiment_folder,f"{self.model_name}_train.log")
        return train_log_path #train_log_folder,

    def get_patch_folders(self,dataset_folder,dataset_part):
        ### PATCH SAVE DIRECTORIES
        patch_folder_base = os.path.join(dataset_folder,dataset_part,f"patches_{self.patch_size}_recognition")
        img_patch_folder = f"{patch_folder_base}/images"
        img_patch_orthogonal_folder = f"{patch_folder_base}/orthogonal_images"
        img_patch_orthogonal_zoomed_folder = f"{patch_folder_base}/orthogonal_zoomed_images"
        label_patch_folder = f"{patch_folder_base}/labels"

        folders = [ dataset_folder,
                    patch_folder_base,
                    img_patch_folder,
                    img_patch_orthogonal_folder,
                    img_patch_orthogonal_zoomed_folder,
                    label_patch_folder]


        ## IF THEY DO NOT EXIST, WRITE THEN
        for folder in folders:
            folder_ok = self.create_folder(folder)
            if not folder_ok:
                return 0
        return folders

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

if __name__ == '__main__':

    model_name = 'custom_0'
    exp_no = 0
    patch_size=128
    settings = Settings(model_name,exp_no,patch_size)()
    print(settings)
