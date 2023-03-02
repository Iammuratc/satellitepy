import torch
from src.settings.utils import create_folder, get_project_folder
from src.data.cutout.geometry import BBox
import cv2
from src.data.cutout.tools import Tools as CutoutTools
from mmrotate.core.bbox import rbbox_overlaps
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.settings.utils import get_logger
# import shutil
## Remove save_folder add it to settings


class Tools(object):
    """docstring for Tools"""
    def __init__(self, settings,save_folder):
        super(Tools, self).__init__()
        self.settings = settings
        self.set_cutout_save_folder(save_folder)
        self.logger = get_logger(file=os.path.join(save_folder,'testing.log'))


    def set_cutout_save_folder(self,save_folder):
        self.binary_mmrotate_cutout_folder = os.path.join(save_folder,'binary')
        assert create_folder(self.binary_mmrotate_cutout_folder)
        self.binary_mmrotate_cutout_original_images_folder = os.path.join(self.binary_mmrotate_cutout_folder,'images_original')
        assert create_folder(self.binary_mmrotate_cutout_original_images_folder)
        self.binary_mmrotate_cutout_orthogonal_images_folder = os.path.join(self.binary_mmrotate_cutout_folder,'images_orthogonal')
        assert create_folder(self.binary_mmrotate_cutout_orthogonal_images_folder)
        self.binary_mmrotate_cutout_labels_folder = os.path.join(self.binary_mmrotate_cutout_folder,'labels')
        assert create_folder(self.binary_mmrotate_cutout_labels_folder)
        self.binary_mmrotate_cutout_padded_orthogonal_images_folder = os.path.join(self.binary_mmrotate_cutout_folder,'images_padded_orthogonal')
        assert create_folder(self.binary_mmrotate_cutout_padded_orthogonal_images_folder)

    def save_cutouts_from_mmrotate_results(self):
        # self.set_cutout_save_folder(save_folder)

        # multiclass_mmrotate_cutout_folder = os.path.join(save_folder,'multiclass')
        # assert create_folder(multiclass_mmrotate_cutout_folder)
        # multiclass_mmrotate_cutout_images_folder = os.path.join(multiclass_mmrotate_cutout_folder,'images')
        # assert create_folder(multiclass_mmrotate_cutout_images_folder)
        # multiclass_mmrotate_cutout_labels_folder = os.path.join(multiclass_mmrotate_cutout_folder,'labels')
        # assert create_folder(multiclass_mmrotate_cutout_labels_folder)

        original_results_folder = self.settings['results']['original_folder']
        image_folder = self.settings['original']['image_folder']

        cutout_settings = {'tasks':'bbox'}
        cutout_tools = CutoutTools(cutout_settings)

        # binary_cutout_file = open(binary_mmrotate_cutout_file_path,'a+')

        for file_name_i, file_name in enumerate(os.listdir(original_results_folder)):
            print(file_name)
            file_name_no_ext = os.path.splitext(file_name)[0]
            results_path = os.path.join(original_results_folder,file_name)
            with open(results_path,'r') as f:
                result = json.load(f)

            bboxes_gt = [BBox(corners=bbox_corners) for bbox_corners in result['all_gt']['bboxes']]
            bboxes_binary_mmrotate = [BBox(corners=bbox_corners) for bbox_corners in result['det_binary_mmrotate']['bboxes']]
            bboxes_multiclass_mmrotate = [BBox(corners=bbox_corners) for bbox_corners in result['det_multiclass_mmrotate']['bboxes']]
            
            # ious = rbbox_overlaps(torch.from_numpy(class_bboxes[:, :5]).float(), torch.from_numpy(gt_bboxes).float())
            ious_binary_mmrotate = rbbox_overlaps(torch.FloatTensor([my_bbox.params for my_bbox in bboxes_binary_mmrotate]), torch.FloatTensor([my_bbox.params for my_bbox in bboxes_gt]))
            # print(ious_binary_mmrotate)
            label_dict = {
                'instance':{'name':None},
                'iou_score':None,
                'det_binary_score':None,
                'bbox_corners_gt':None,
                'bbox_corners_det':None
            }
            img = cutout_tools.get_original_image(os.path.join(image_folder,f'{file_name_no_ext}.tif'))
            # if len(bboxes_gt) > len(bboxes_binary_mmrotate):
            # det_bboxes_binary = []          
            for i,iou in enumerate(ious_binary_mmrotate):
                # ROW: detected bboxes
                # COL: gt bboxes
                # if result['det_binary_mmrotate']['scores'][i]<0.2:
                #     continue
                # try:
                cutout_file_name = f'{file_name_no_ext}_{i}'
                det_binary_score = result['det_binary_mmrotate']['scores'][i]
                # try:
                bbox_ind_gt = np.argmax(iou)
                # det_bboxes_binary.append(int(bbox_ind_gt))
                iou_score = iou[bbox_ind_gt]
                # except:
                #     continue
                det_bbox_gt = bboxes_gt[bbox_ind_gt].corners # cutout to be classified by the mmclass models
                # print(det_bbox_gt)

                ### LABEL
                # if iou_score>iou_threshold:
                instance_name_gt = result['all_gt']['instance_names'][bbox_ind_gt]
                # else:
                #     # print('Here')
                #     instance_name_gt = 'Background'

                cutout_dict = cutout_tools.init_cutout_dict(
                    instance_name='Airplane',
                    img_path=None,
                    mask_path=None)
                # print(label_dict)

                ### IMAGE
                cutout_dict = cutout_tools.set_cutout_params(cutout_dict, img, det_bbox_gt, mask=None)
                original_image_file_path = os.path.join(self.binary_mmrotate_cutout_original_images_folder,f'{cutout_file_name}.png')
                cv2.imwrite(original_image_file_path,cutout_dict['original_cutout']['img'])

                ### LABEL
                label_dict['instance']['name']=instance_name_gt
                label_dict['iou_score']=float(iou_score)
                label_dict['det_binary_score']=det_binary_score
                label_dict['bbox_corners_gt'] = cutout_dict['original_cutout']['bbox']['corners'].tolist()
                label_dict['bbox_corners_det'] = bboxes_binary_mmrotate[i].corners.astype(int).tolist()
                # print(label_dict)
                with open(os.path.join(self.binary_mmrotate_cutout_labels_folder,f'{cutout_file_name}.json'),'w+') as f:
                    json.dump(label_dict, f, indent=4)
                # orthogonal_image_file_path = os.path.join(binary_mmrotate_cutout_orthogonal_images_folder,f'{cutout_file_name}.png')
                # cv2.imwrite(orthogonal_image_file_path,cutout_dict['orthogonal_cutout']['img'])
                # fig,ax = plt.subplots(1)
                # ax.imshow(img)
                # BBox.plot_bbox(det_bbox_gt,ax,c='b',instance_name=None)
                # ax.imshow(cutout_dict['original_cutout']['img'])
                # plt.show()
                # break
            # break
            # label_dict = {
            #     'instance':{'name':None},
            #     'iou_score':None,
            #     'det_multiclass_score':None,
            # }

    def align_cutouts(self):
        from src.settings.orientation import SettingsOrientation
        from src.classifier.orientation import ClassifierOrientation 

        project_folder = get_project_folder()

        my_settings = SettingsOrientation(
            model_config=os.path.join(project_folder,'experiments','exp_11','settings.json'),   
            template_image_path=os.path.join(project_folder,'data','fair1m','airliner_template.png'))()


        classifier = ClassifierOrientation(my_settings)
        # self.set_cutout_save_folder(save_folder)
        generator = self.get_original_cutout_generator()
        classifier.fix_orientation(generator=generator,save_folder=self.binary_mmrotate_cutout_orthogonal_images_folder)

    def get_original_cutout_generator(self):
        file_names = os.listdir(self.binary_mmrotate_cutout_original_images_folder)
        # file_names.sort()
        for file_name in file_names:
            print(file_name)
            ### BBOX
            file_name_no_ext = os.path.splitext(file_name)[0]
            bbox_path = os.path.join(self.binary_mmrotate_cutout_labels_folder,f"{file_name_no_ext}.json")
            with open(bbox_path,'r') as f:
                bbox_labels = json.load(f)
            bbox_corners = bbox_labels['bbox_corners_gt']

            ### CUTOUT
            image_path = os.path.join(self.binary_mmrotate_cutout_original_images_folder,file_name)
            img = cv2.imread(image_path,1)

            # print(img.shape)
            # print(bbox_corners)

            ### PLOT
            # fig, ax = plt.subplots(1)
            # # ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # ax.imshow(cv2.cvtColor(img_ort,cv2.COLOR_BGR2RGB))
            # my_bbox = BBox(corners=bbox_corners)
            # # my_bbox.plot_bbox(corners=bbox_corners, ax=ax, c='b', s=15, instance_name=None)
            # my_bbox.plot_bbox(corners=bbox_ort, ax=ax, c='b', s=15, instance_name=None)
            # plt.show()
            yield img, bbox_corners, file_name#, img_ort, bbox_corners_ort

    def classify_cutouts(self,iou_th):
        from src.classifier.classification import ClassifierClassification
        from src.data.dataset.classification import DatasetClassification
        from src.transforms import Normalize, ToTensor, AddAxis, HorizontalFlip
        from torchvision.transforms import Compose
        
        with open(os.path.join(get_project_folder(),'data_settings','fair1m_settings.json'),'r') as f:
            data_settings=json.load(f)
        # with open(os.path.join(get_project_folder(),'experiments','efficientnet_b3_padded_orthogonal_cutout','exp_settings.json'),'r') as f:
        with open(os.path.join(get_project_folder(),'experiments','efficientnet_b3_padded_orthogonal_cutout','exp_settings.json'),'r') as f:
            exp_settings=json.load(f)

        dataset = DatasetClassification(
            exp_settings=exp_settings,
            data_settings=data_settings,
            dataset_part='val',
            transform=Compose([ToTensor(), Normalize(task='classification')],),
            image_folder=self.get_iou_filtered_image_folder(iou_th,cutout_config='orthogonal'),
            label_folder=self.binary_mmrotate_cutout_labels_folder
            )

        # print(next(iter(dataset)))

        my_classifier = ClassifierClassification(data_settings,exp_settings)
        loader_val = my_classifier.get_loader(dataset, shuffle=False, batch_size=exp_settings['training']['batch_size'])
        conf_matrix = my_classifier.test(loader_val)
        precision, recall = self.get_precision_recall(conf_matrix)
        for i, instance_name in enumerate(list(data_settings['instance_names'].keys())):
            pre = precision[i]*100
            print(f'{instance_name},{pre:.2f}')

    def get_iou_filtered_image_folder(self,iou_th,cutout_config):
        iou_folder_name =  str(iou_th).replace('.', "")
        image_folder = os.path.join(self.binary_mmrotate_cutout_folder,f'images_{cutout_config}_iuo_{iou_folder_name}')
        assert create_folder(image_folder)
        if os.listdir(image_folder)==[]: # os.path.exists(image_folder) and
            for file_name in os.listdir(self.binary_mmrotate_cutout_labels_folder):
                with open(os.path.join(self.binary_mmrotate_cutout_labels_folder,file_name)) as f:
                    label_dict = json.load(f)
                print(label_dict)
                if label_dict['iou_score']>iou_th:
                    image_name = os.path.splitext(file_name)[0]
                    # src = os.path.join(self.binary_mmrotate_cutout_original_images_folder,f'{image_name}.png')
                    src = os.path.join(self.binary_mmrotate_cutout_padded_orthogonal_images_folder,f'{image_name}.png')
                    # print(src)
                    dst = os.path.join(image_folder,f'{image_name}.png')
                    os.symlink(src,dst)
        return image_folder


    def pad_cutouts(self):
        from src.utilities import resize_cutouts_by_padding

        resize_cutouts_by_padding(
            image_folder=self.binary_mmrotate_cutout_orthogonal_images_folder,
            save_folder=self.binary_mmrotate_cutout_padded_orthogonal_images_folder,
            patch_size=128)

    def get_precision_recall(self,conf_matrix):
        precision = np.zeros(shape=(conf_matrix.shape[0]))
        recall = np.zeros(shape=(conf_matrix.shape[0]))
        # average_precision = np.zeros(shape=(conf_matrix.shape[0]))

        for i in range(conf_matrix.shape[0]): ## Row is GT
            tp = 0
            fp = 0
            fn = 0
            for j in range(conf_matrix.shape[1]):
                # Precision
                if i == j:
                    tp = conf_matrix[i,j]
                else:
                    fn += conf_matrix[i,j]
                    fp += conf_matrix[j,i]
                precision[i] = tp/(tp+fp)
                recall[i] = tp/(tp+fn)
        return precision, recall
        # print(precision)
        # print(recall)
        # average_precision = precision*recall/0.1
        # print(average_precision)
        # print(np.nanmean(average_precision))
