from .utils import get_project_folder, get_logger
import os

class SettingsMMRotate:
    """docstring for MMRotate"""
    def __init__(self,
        exp_name,
        exp_root='mmrotate/work_dirs',
        test_pkl='test.pkl',
        checkpoint='latest.pth'):
        
        # super(MMRotate, self).__init__()
        self.exp_root = exp_root
        self.exp_name = exp_name
        self.test_pkl = test_pkl
        self.checkpoint=checkpoint
        self.config=f"{exp_name}.py"

    def __call__(self):
        return self.get_settings()

    def get_settings(self):

        ## EXPERIMENT FOLDER
        project_folder = get_project_folder()
        exp_folder = os.path.join(project_folder,self.exp_root,self.exp_name)

        ## MMROTATE RESULTS (PICKLE)
        test_pkl_path = os.path.join(exp_folder,self.test_pkl)
        assert os.path.exists(test_pkl_path)
        ## MMROTATE RESULTS (CSV)
        test_pkl_name,_=os.path.splitext(self.test_pkl)
        test_csv_path = os.path.join(exp_folder,f"{test_pkl_name}.csv")
        ## MMROTATE RESULT IMAGES FOLDER
        test_images_path = os.path.join(exp_folder,'results')

        ## MMROTATE CHECKPOINT
        checkpoint_path = os.path.join(exp_folder,self.checkpoint)

        ## MMROTATE CONFIG
        config_path = os.path.join(exp_folder,self.config)

        ## LOGGING
        log_name = self.exp_name
        log_file = os.path.join(exp_folder,f"src_{log_name}.log")

        settings = {
            'logger':get_logger(name=log_name,file=log_file),
            'exp_folder':exp_folder,
            'test_pkl_path':test_pkl_path,
            'test_csv_path':test_csv_path,
            'test_images_path':test_images_path,
            'config_path':config_path,
            'checkpoint_path':checkpoint_path
            }
        return settings