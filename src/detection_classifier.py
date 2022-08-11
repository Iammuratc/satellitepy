import os
import yaml



class DetectionClassifier:
    def __init__(self,settings):

        self.settings_utils=settings
        self.settings=self.settings_utils()

    def train(self):

        ## YOLO train command
        # !python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
        train_py = self.settings['training']['yolo']['train_py']
        img_size = self.settings['patch']['size']
        batch = self.settings['training']['batch_size']
        epochs = self.settings['training']['epochs']
        data_yaml,hyp_yaml = self.get_yolo_yaml_files()
        weights = self.settings['training']['yolo']['weights']
        weights_yaml = self.settings['training']['yolo']['weights_yaml']
        patience = self.settings['training']['patience']

        experiment_folder = self.settings['experiment']['folder']
        # resume = self.settings['training']['resume']
        # --resume {resume}\
        train_cmd = f'python3 {train_py} --project {experiment_folder} --exist-ok\
                        --imgsz {img_size} --batch {batch} --epochs {epochs} --data {data_yaml} \
                        --hyp {hyp_yaml} --weights {weights} --cfg {weights_yaml} --patience {patience} --cache'
        os.system(train_cmd)

    def validate(self,dataset_part):
        test_py = self.settings['testing']['yolo']['test_py']
        img_size = self.settings['patch']['size']
        batch = self.settings['training']['batch_size']
        weights = self.settings['training']['yolo']['weights']
        data_yaml,_ = self.get_yolo_yaml_files()
        experiment_folder = self.settings['experiment']['folder']

        test_cmd = f'python3 {test_py} --project {experiment_folder} --task {dataset_part}\
                    --imgsz {img_size} --batch {batch} --data {data_yaml} --weights {weights}'
        os.system(test_cmd)

    def get_yolo_yaml_files(self):
        # /home/murat/Projects/airplane_recognition/data/Gaofen/train/detection/yolov5.yaml
        # train: /home/murat/Projects/airplane_detection/DATA/Gaofen/train/patches_512/images
        # val: /home/murat/Projects/airplane_detection/DATA/Gaofen/val/patches_512/images

        # nc: 1
        # names: ['airplane']

        # /home/murat/Projects/airplane_recognition/data/Gaofen/train/detection/hyp_conf.yaml

        ### DATA YAML
        data_yaml_path = self.settings['training']['yolo']['data_yaml']
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, "r") as stream:
                try:
                    data_yaml = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:

            train_img_patch_folder, test_img_patch_folder, val_img_patch_folder = self.get_patch_folders()
            data_yaml = {   'train':train_img_patch_folder,
                            'val':val_img_patch_folder,
                            'test':test_img_patch_folder,
                            'nc':1,
                            'names':['airplanes']}
            with open(data_yaml_path,'w+') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

        ### HYPERPARAMETER CONFIG YAML
        hyp_yaml_path = self.settings['training']['yolo']['hyp_config_yaml']
        if not os.path.exists(hyp_yaml_path):
            hyp_yaml = {
                    'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'lrf': 0.01,  # final OneCycleLR learning rate (lr0 * lrf)
                    'momentum': 0.937,  # SGD momentum/Adam beta1
                    'weight_decay': 0.0005,  # optimizer weight decay 5e-4
                    'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
                    'warmup_momentum': 0.8,  # warmup initial momentum
                    'warmup_bias_lr': 0.1,  # warmup initial bias lr
                    'box': 0.05,  # box loss gain
                    'cls': 0.3,  # cls loss gain
                    'cls_pw': 1.0,  # cls BCELoss positive_weight
                    'obj': 0.7,  # obj loss gain (scale with pixels)
                    'obj_pw': 1.0,  # obj BCELoss positive_weight
                    'iou_t': 0.20,  # IoU training threshold
                    'anchor_t': 4.0,  # anchor-multiple threshold
                    'anchors': 3,  # anchors per output layer (0 to ignore)
                    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
                    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
                    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
                    'degrees': 10.0,  # image rotation (+/- deg) # 0.0
                    'translate': 0.0,  # image translation (+/- fraction)
                    'scale': 0.9,  # image scale (+/- gain)
                    'shear': 10.0,  # image shear (+/- deg) # 0.0
                    'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
                    'flipud': 0.5,  # image flip up-down (probability)
                    'fliplr': 0.5,  # image flip left-right (probability)
                    'mosaic': 0.0,  # image mosaic (probability)
                    'mixup': 0.1,  # image mixup (probability)
                    'copy_paste': 0.1,  # segment copy-paste (probability)}
                    }

            with open(hyp_yaml_path,'w+') as f:
                yaml.dump(hyp_yaml, f, default_flow_style=False, allow_unicode=True)

        return data_yaml_path, hyp_yaml_path

    def get_patch_folders(self):
        split_ratio = self.settings['training']['split_ratio']

        train_img_patch_folder = self.settings['patch']['train']['img_patch_folder']
        test_img_patch_folder = self.settings['patch']['test']['img_patch_folder']
        val_img_patch_folder = self.settings['patch']['val']['img_patch_folder']

        train_label_patch_folder = self.settings['patch']['train']['label_patch_yolo_folder']
        test_label_patch_folder = self.settings['patch']['test']['label_patch_yolo_folder']
        val_label_patch_folder = self.settings['patch']['val']['label_patch_yolo_folder']

        if split_ratio:
            ### SPLIT ORIGINAL DATASET PARTS
            # Split image paths according to the split_ratio
            # Sort image paths, because there are overlaps among some images, check out sequences.json
    
            ### IMAGE PATHS
            train_img_paths = self.get_full_paths(train_img_patch_folder)
            test_img_paths = self.get_full_paths(test_img_patch_folder)
            val_img_paths = self.get_full_paths(val_img_patch_folder)

            ### LABEL PATHS         
            train_label_paths = self.get_full_paths(train_label_patch_folder)
            test_label_paths = self.get_full_paths(test_label_patch_folder)
            val_label_paths = self.get_full_paths(val_label_patch_folder)
    
            ### ALL PATHS
            all_img_paths = train_img_paths + test_img_paths + val_img_paths
            all_label_paths = train_label_paths + test_label_paths + val_label_paths

            all_len = len(all_img_paths)
            train_len = int(all_len*split_ratio[0])
            test_len = int(all_len*split_ratio[1])
            val_len = int(all_len - train_len - test_len)

            ### SPLIT PATHS
            split_train_img_paths = all_img_paths[:train_len]
            split_test_img_paths = all_img_paths[train_len:train_len+test_len]
            split_val_img_paths = all_img_paths[train_len+test_len:]

            split_train_label_paths = all_label_paths[:train_len]
            split_test_label_paths = all_label_paths[train_len:train_len+test_len]
            split_val_label_paths = all_label_paths[train_len+test_len:]

            # ### SYMLINK FOLDERS
            train_img_symlink_folder = os.path.join(self.settings['experiment']['folder'],'data_symlinks','train','images')
            test_img_symlink_folder = os.path.join(self.settings['experiment']['folder'],'data_symlinks','test','images')
            val_img_symlink_folder = os.path.join(self.settings['experiment']['folder'],'data_symlinks','val','images')

            train_label_symlink_folder = os.path.join(self.settings['experiment']['folder'],'data_symlinks','train','labels')
            test_label_symlink_folder = os.path.join(self.settings['experiment']['folder'],'data_symlinks','test','labels')
            val_label_symlink_folder = os.path.join(self.settings['experiment']['folder'],'data_symlinks','val','labels')

            self.create_symlink_folder(folder=train_img_symlink_folder,paths=split_train_img_paths)
            self.create_symlink_folder(folder=test_img_symlink_folder,paths=split_test_img_paths)
            self.create_symlink_folder(folder=val_img_symlink_folder,paths=split_val_img_paths)

            self.create_symlink_folder(folder=train_label_symlink_folder,paths=split_train_label_paths)
            self.create_symlink_folder(folder=test_label_symlink_folder,paths=split_test_label_paths)
            self.create_symlink_folder(folder=val_label_symlink_folder,paths=split_val_label_paths)



            return [train_img_symlink_folder,
                    test_img_symlink_folder,
                    val_img_symlink_folder]#,
                    # train_label_symlink_folder,
                    # test_label_symlink_folder,
                    # val_label_symlink_folder]
        else:
            return [train_img_patch_folder,
                    test_img_patch_folder,
                    val_img_patch_folder]#,
                    # train_label_patch_folder,
                    # test_label_patch_folder,
                    # val_label_patch_folder]


    def get_full_paths(self,folder):
        sorted_files = os.listdir(folder)
        sorted_files.sort(key=lambda f: (int(f.split('_')[0]),int(f.split('_')[2]),int(f.split('_')[4].split('.')[0]))) # EX: 1_x_0_y_0.json
        return [os.path.join(folder,file) for file in sorted_files]

    def create_symlink_folder(self,folder,paths):

        folder_ok = self.settings_utils.create_folder(folder)
        if not folder_ok:
            return 0

        for path in paths:
            _, file = os.path.split(path)
            file_name, ext = os.path.splitext(file)
            symlink_path = os.path.join(folder,f"{file_name}_symlink{ext}")
            try:
                os.symlink(path,symlink_path)
            except FileExistsError:
                continue

if __name__ == '__main__':
    from settings import SettingsDetection

    ## EXPERIMENT
    exp_no = 6
    exp_name = 'First yolov5l detection trial'
    model_name = 'yolov5l'

    ### PATCH
    patch_size = 512
    split_ratio = [0.802,0.1,0.098]

    ### TRAINING
    batch_size = 15
    epochs = 50
    patience= 10
    # resume=True



    settings = SettingsDetection(update=True,
                                model_name=model_name,
                                exp_no=exp_no,
                                exp_name=exp_name,
                                patch_size=patch_size,
                                split_ratio=split_ratio,
                                batch_size=batch_size,
                                epochs=epochs,
                                patience=patience,
                                # resume=resume
                                )


    classifier = DetectionClassifier(settings)
    # classifier.train()
    classifier.validate(dataset_part='test')

    # import os
    # import cv2
    # img_path = '/home/murat/Projects/airplane_recognition/data/Gaofen/train/detection/patches_512/images/1_x_0_y_0.png'

    # img_path_link = 'temp.png'

    # os.symlink(img_path,img_path_link)

    # img_orig = cv2.imread(img_path)

    # print(img_orig.shape)
    # img_link = cv2.imread(img_path_link)

    # print(img_link.shape)