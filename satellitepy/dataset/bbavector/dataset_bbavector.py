import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import hashlib

from satellitepy.dataset.bbavector.utils import Utils
from satellitepy.data.labels import read_label, satellitepy_labels_empty
from satellitepy.data.utils import get_satellitepy_dict_values, get_task_dict #, merge_satellitepy_task_values
from satellitepy.utils.path_utils import get_file_paths, zip_matched_files


class BBAVectorDataset(Dataset):
    # def __init__(self, data_dir, input_h=None, input_w=None, down_ratio=None):
    def __init__(self,
        in_image_folder,
        in_label_folder,
        in_label_format,
        tasks,
        input_h,
        input_w,
        down_ratio,
        augmentation = False,
        validate_dataset = True,
        K = 1000,
        random_seed=12):
        super(BBAVectorDataset, self).__init__()

        self.utils = Utils(tasks, input_h, input_w, down_ratio, K=K)
        # self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        # self.cat_ids = task_dict
        self.tasks = tasks
        # self.img_ids = self.load_img_ids()
        # self.image_path = os.path.join(data_dir, 'images')
        # self.label_path = os.path.join(data_dir, 'labelTxt')
        self.items = []
        self.augmentation = augmentation
        self.random_seed = random_seed
        self.testing = not in_label_folder

        self.grouped_tasks = []
        for t in self.tasks:
            if t not in ["obboxes", "hbboxes", "masks"]:
                task_dict = get_task_dict(t)
                if 'min' in task_dict.keys() and 'max' in task_dict.keys():
                    self.grouped_tasks.append("reg_" + t)
                else:
                    self.grouped_tasks.append("cls_" + t)
            else:
                self.grouped_tasks.append(t)

        if not in_label_folder:
            for img_path in get_file_paths(in_image_folder):
                self.items.append((img_path, None, None))
        else:
            if validate_dataset:
                total = len(os.listdir(in_image_folder))
                removed = 0
                pbar = tqdm(zip_matched_files(in_image_folder,in_label_folder), total=total, desc="validating data")

                for img_path, label_path in pbar:
                    hash_str = str(img_path) + str(label_path) + str(self.random_seed)
                    hash_bytes = hashlib.sha256(bytes(hash_str, "utf-8")).digest()[:4]
                    np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
                    image = cv2.imread(img_path.absolute().as_posix())
                    labels = read_label(label_path,in_label_format)
                    del labels["masks"]
                    image_h, image_w, c = image.shape
                    annotation = self.preapare_annotations(labels, image_w, image_h)#, img_path)
                    image, annotation = self.utils.data_transform(image, annotation, self.augmentation)

                    if self.all_tasks_available(annotation):
                        self.items.append((img_path, label_path, in_label_format))
                    else:
                        removed += 1
                        pbar.set_description(f"validating data (removed: {removed})")
            else:
                for img_path, label_path in zip_matched_files(in_image_folder, in_label_folder):
                    self.items.append((img_path, label_path, in_label_format))

    def __len__(self):
        return len(self.items)

    def all_tasks_available(self, annotation: dict):
        # basically checks, if labels had a bad format, or annotations are missing after random crop
        existing = list(annotation.keys())
        for k in self.grouped_tasks:
            if k not in existing: return False
        return True

    def __getitem__(self, idx):
        ### Image
        img_path, label_path, label_format = self.items[idx]
        image = cv2.imread(img_path.absolute().as_posix())
        image_h, image_w, c = image.shape

        if self.testing:
            return {
                "input": torch.from_numpy(image / 255).permute((2, 0, 1)),
                "img_path": str(img_path),
                "img_w": image_w,
                "img_h": image_h
            }

        ### Labels
        labels = read_label(label_path,label_format)
        hash_str = str(img_path) + str(label_path) + str(self.random_seed)
        hash_bytes = hashlib.sha256(bytes(hash_str, "utf-8")).digest()[:4]
        np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
        annotation = self.preapare_annotations(labels, image_w, image_h)#, img_path)
        # print(annotation['masks'].shape)

        image, annotation = self.utils.data_transform(image, annotation, self.augmentation)
        # print(annotation['masks'].shape)
        data_dict = self.utils.generate_ground_truth(image, annotation)
        data_dict['img_path']=str(img_path)
        data_dict['label_path']=str(label_path)
        data_dict['img_w']=image_w
        data_dict['img_h']=image_h

        return data_dict

    def prepare_masks(self, labels, image_width, image_height):#, image_file = None):
        if "masks" not in labels:
            return None
        masks = np.zeros((image_height, image_width))

        for val in labels["masks"]:
            if val is None:
                continue
            else:
                m_x, m_y = val
            if len(m_x) == 0 or len(m_y) == 0:
                continue
            m_x = np.array(m_x).clip(0, image_width - 1)
            m_y = np.array(m_y).clip(0, image_height - 1)
            masks[m_y, m_x] = 1.0

        if np.count_nonzero(masks) == 0:
            return None

        return masks

    def visualize_masks(self, image, masks, labels):
        #debug
        mask_image = np.array(image)
        image_h, image_w, _ = image.shape
        mask = np.zeros((image_h, image_w, 1), dtype=np.uint8)
        for m in np.moveaxis(masks, -1, 0):
            test_m = m.astype(np.uint8)[:, :, np.newaxis]
            mask = cv2.bitwise_or(mask, test_m)

        mask_image = cv2.bitwise_and(mask_image, mask_image, mask=mask)
        vis_image = np.concatenate((mask_image, image), axis=1)
        title = ",".join(list(set([value for value in get_satellitepy_dict_values(labels,self.task)])))
        cv2.imshow(title, vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def preapare_annotations(self, labels, image_w, image_h):#, img_path):
        annotation = {}
        for t in self.tasks:
            if t in ["obboxes", "hbboxes"]:
                annotation[t] = np.asarray(get_satellitepy_dict_values(labels, t))
            elif t == "masks":
                annotation[t] = self.prepare_masks(labels , image_w, image_h)#, str(img_path))
            else:
                task_dict = get_task_dict(t)

                if 'min' in task_dict.keys() and 'max' in task_dict.keys():
                    values = np.asarray(get_satellitepy_dict_values(labels, t))
                    max, min = task_dict["max"], task_dict["min"]
                    normalized = [
                        (val - min) / (max - min) if val is not None else None
                        for val in values
                    ]
                    annotation["reg_" + t] = normalized
                else:
                    annotation["cls_" + t] = np.asarray([
                        task_dict[value] if value is not None else None
                        for value in get_satellitepy_dict_values(labels,t)
                    ])
        return annotation


    
    # def load_img_ids(self):
    #     if self.phase == 'train':
    #         image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')
    #     else:
    #         image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
    #     assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
    #     with open(image_set_index_file, 'r') as f:
    #         lines = f.readlines()
    #     image_lists = [line.strip() for line in lines]
    #     return image_lists

    # def load_image(self, index):
    #     img_id = self.img_ids[index]
    #     imgFile = os.path.join(self.image_path, img_id+'.png')
    #     assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
    #     img = cv2.imread(imgFile)
    #     return img

    # def load_annoFolder(self, img_id):
    #     return os.path.join(self.label_path, img_id+'.txt')

    # def load_annotation(self, index):
    #     image = self.load_image(index)
    #     h,w,c = image.shape
    #     valid_pts = []
    #     valid_cat = []
    #     valid_dif = []
    #     with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
    #         for i, line in enumerate(f.readlines()):
    #             obj = line.split(' ')  # list object
    #             if len(obj)>8:
    #                 x1 = min(max(float(obj[0]), 0), w - 1)
    #                 y1 = min(max(float(obj[1]), 0), h - 1)
    #                 x2 = min(max(float(obj[2]), 0), w - 1)
    #                 y2 = min(max(float(obj[3]), 0), h - 1)
    #                 x3 = min(max(float(obj[4]), 0), w - 1)
    #                 y3 = min(max(float(obj[5]), 0), h - 1)
    #                 x4 = min(max(float(obj[6]), 0), w - 1)
    #                 y4 = min(max(float(obj[7]), 0), h - 1)
    #                 # TODO: filter small instances
    #                 xmin = max(min(x1, x2, x3, x4), 0)
    #                 xmax = max(x1, x2, x3, x4)
    #                 ymin = max(min(y1, y2, y3, y4), 0)
    #                 ymax = max(y1, y2, y3, y4)
    #                 if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
    #                     valid_pts.append([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
    #                     valid_cat.append(self.cat_ids[obj[8]])
    #                     valid_dif.append(int(obj[9]))
    #     f.close()
    #     annotation = {}
    #     annotation['pts'] = np.asarray(valid_pts, np.float32)
    #     annotation['cat'] = np.asarray(valid_cat, np.int32)
    #     annotation['dif'] = np.asarray(valid_dif, np.int32)
    #     # pts0 = np.asarray(valid_pts, np.float32)
    #     # img = self.load_image(index)
    #     # for i in range(pts0.shape[0]):
    #     #     pt = pts0[i, :, :]
    #     #     tl = pt[0, :]
    #     #     tr = pt[1, :]
    #     #     br = pt[2, :]
    #     #     bl = pt[3, :]
    #     #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
    #     #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
    #     #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
    #     #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
    #     #     cv2.putText(img, '{}:{}'.format(valid_dif[i], self.category[valid_cat[i]]), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
    #     #                 (0, 0, 255), 1, 1)
    #     # cv2.imshow('img', np.uint8(img))
    #     # k = cv2.waitKey(0) & 0xFF
    #     # if k == ord('q'):
    #     #     cv2.destroyAllWindows()
    #     #     exit()
    #     return annotation


    # def merge_crop_image_results(self, result_path, merge_path):
    #     mergebypoly(result_path, merge_path)

        # self.category = ['plane',
        #                  'baseball-diamond',
        #                  'bridge',
        #                  'ground-track-field',
        #                  'small-vehicle',
        #                  'large-vehicle',
        #                  'ship',
        #                  'tennis-court',
        #                  'basketball-court',
        #                  'storage-tank',
        #                  'soccer-ball-field',
        #                  'roundabout',
        #                  'harbor',
        #                  'swimming-pool',
        #                  'helicopter'
        #                  ]
        # self.color_pans = [(204,78,210),
        #                    (0,192,255),
        #                    (0,131,0),
        #                    (240,176,0),
        #                    (254,100,38),
        #                    (0,0,255),
        #                    (182,117,46),
        #                    (185,60,129),
        #                    (204,153,255),
        #                    (80,208,146),
        #                    (0,0,204),
        #                    (17,90,197),
        #                    (0,255,255),
        #                    (102,255,102),
        #                    (255,255,0)]
        
