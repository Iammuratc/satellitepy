import os
import json


## TODO: move all the training paths in Recognition to settings

class Settings(object):
    """docstring for Settings"""
    def __init__(self,  model_name=None,
                        exp_no=None,
                        patch_size=None,
                        batch_size=None,
                        epochs=None,
                        hot_encoding=None,
                        update=True):
        super(Settings, self).__init__()
        self.update=update
        self.model_name = model_name
        self.exp_no = exp_no
        self.patch_size = patch_size

        ### TRAINING HYPERPARAMETERS
        self.hot_encoding=hot_encoding
        self.batch_size=batch_size
        self.epochs=epochs
    
        self.project_folder = self.get_project_folder()

    def __call__(self):
        return self.get_settings()

    def get_settings(self):
        ### SETTINGS PATH
        settings_folder = os.path.join(self.project_folder,"settings")
        folder_ok = self.create_folder(settings_folder)
        if not folder_ok:
            return 0

        settings_exp_folder = os.path.join(self.project_folder,"settings",f"exp_{self.exp_no}")
        folder_ok = self.create_folder(settings_exp_folder)
        if not folder_ok:
            return 0

        settings_path = os.path.join(settings_exp_folder,f"settings.json")

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
            'exp_no':self.exp_no,
            'model': {
                'name':self.model_name,
                'path':self.get_model_path(),
            },

            'dataset': {
                'folder':dataset_folder,
                'name':dataset_name,
                'id': dataset_id
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
                'hot_encoding':self.hot_encoding,
                'batch_size':self.batch_size,
                'epochs':self.epochs,
                # 'log_folder':
                'log_path':self.get_train_log_path(),
            }
        }

        with open(settings_path,'w+') as f:
            json.dump(settings,f,indent=4)
        
        return settings

    def get_project_folder(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_model_path(self):
        ### BINARIES FOLDER
        binaries_folder = os.path.join(self.project_folder,'binaries')
        folder_ok = self.create_folder(binaries_folder)
        if not folder_ok:
            return 0

        ### MODEL FOLDER
        model_folder = os.path.join(binaries_folder,f'exp_{self.exp_no}')
        folder_ok = self.create_folder(model_folder)
        if not folder_ok:
            return 0

        ### ASK USER TO CREATE THE FOLDER
        model_path = os.path.join(model_folder,f"{self.model_name}.pth")

        return model_path

    def get_train_log_path(self):
        ### BINARIES FOLDER
        log_folder = os.path.join(self.project_folder,'logs')
        folder_ok = self.create_folder(log_folder)
        if not folder_ok:
            return 0

        ### MODEL FOLDER
        train_log_folder = os.path.join(log_folder,f'exp_{self.exp_no}')
        folder_ok = self.create_folder(train_log_folder)
        if not folder_ok:
            return 0

        train_log_path = os.path.join(train_log_folder,f"{self.model_name}.log")
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
