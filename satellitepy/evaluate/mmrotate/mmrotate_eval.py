import json
import numpy as np
import os
import torch
import mmcv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from mmrotate.core.bbox import rbbox_overlaps
from src.data.cutout.geometry import BBox

class MMRotateEval(object):
    """docstring for MMRotateEval"""
    def __init__(self, settings):
        super(MMRotateEval, self).__init__()
        self.settings = settings
        

    def plot_image(self,file_name):
        ### Get Results
        original_results_folder = self.settings['results']['original_folder']
        results_path = os.path.join(original_results_folder,f"{file_name}.json")
        with open(results_path,'r') as f:
            result = json.load(f)

        ### Get image
        ### ORIGINAL IMAGE FOLDER
        original_image_folder = self.settings['original']['image_folder']
        img = mmcv.imread(os.path.join(original_image_folder,f"{file_name}.tif"))

        ### Plot
        fig,ax =plt.subplots(1)
        ax.set_title('GT TRUTH')
        ax.imshow(img)
        for i_gt,bbox_gt in enumerate(result['all_gt']['bboxes']):
            # print(bbox_gt)
            BBox.plot_bbox(bbox_gt,ax,c='b',instance_name=result['all_gt']['instance_names'][i_gt])

        # for i_det,bbox_det in enumerate(merged_nmsed_patch_results['det_binary_mmrotate']['bboxes']):
        #     BBox.plot_bbox(bbox_det,ax,c='g',instance_name=merged_nmsed_patch_results['original_cutout']['instance_names'][i_det])
        #     BBox.plot_bbox(bbox_det,ax,c='g',instance_name=merged_nmsed_patch_results['orthogonal_cutout']['instance_names'][i_det])

        # for i_det,bbox_det in enumerate(merged_nmsed_patch_results['det_multiclass_mmrotate']['bboxes']):
        #     BBox.plot_bbox(bbox_det,ax,c='r')

        plt.show()


    def get_conf_matrices_original_images(self):
        original_results_folder = self.settings['results']['original_folder']
        instance_names = self.settings['original']['multiclass_instance_names'] + ['Background']
            
        conf_mat_0 = np.zeros(shape=(len(instance_names),len(instance_names))) # BINARY MMROTATE + ORIGINAL CUTOUT
        conf_mat_1 = np.zeros(shape=(len(instance_names),len(instance_names))) # BINARY MMROTATE + ORTHOGONAL CUTOUT
        conf_mat_2 = np.zeros(shape=(len(instance_names),len(instance_names))) # MULTICLASS MMROTATE

        iou_threshold = 0.5
        precisions = []
        # for confidence_score in np.arange(0,1,0.1):
        #     print(f'confidence_score: {confidence_score}')
        confidence_score=0.2
        for file_name_i, file_name in enumerate(os.listdir(original_results_folder)):
            # print(file_name)
            results_path = os.path.join(original_results_folder,file_name)
            with open(results_path,'r') as f:
                result = json.load(f)

            bboxes_gt = [BBox(corners=bbox_corners).params for bbox_corners in result['all_gt']['bboxes']]
            bboxes_binary_mmrotate = [BBox(corners=bbox_corners).params for bbox_corners in result['det_binary_mmrotate']['bboxes']]
            bboxes_multiclass_mmrotate = [BBox(corners=bbox_corners).params for bbox_corners in result['det_multiclass_mmrotate']['bboxes']]
            
            # ious = rbbox_overlaps(torch.from_numpy(class_bboxes[:, :5]).float(), torch.from_numpy(gt_bboxes).float())
            ious_binary_mmrotate = rbbox_overlaps(torch.FloatTensor(bboxes_binary_mmrotate), torch.FloatTensor(bboxes_gt))
            # print(io)

            # if len(bboxes_gt) > len(bboxes_binary_mmrotate):
            det_bboxes_binary = []          
            for i,iou in enumerate(ious_binary_mmrotate):
                # ROW: detected bboxes
                # COL: gt bboxes
                if result['det_binary_mmrotate']['scores'][i]<confidence_score:
                    continue
                try:
                    bbox_ind_gt = np.argmax(iou)
                    det_bboxes_binary.append(int(bbox_ind_gt))
                    iou_score = iou[bbox_ind_gt]
                except:
                    continue

                if iou_score>iou_threshold:
                    instance_name_gt = result['all_gt']['instance_names'][bbox_ind_gt]
                else:
                    # print('Here')
                    instance_name_gt = 'Background'
                instance_name_orthogonal_cutout = result['orthogonal_cutout']['instance_names'][i]
                instance_name_original_cutout = result['original_cutout']['instance_names'][i]

                conf_mat_0[instance_names.index(instance_name_gt),instance_names.index(instance_name_original_cutout)] += 1
                conf_mat_1[instance_names.index(instance_name_gt),instance_names.index(instance_name_orthogonal_cutout)] += 1
                # print(instance_name_gt, instance_name_original_cutout, instance_name_orthogonal_cutout)
            
            ## Process not detected bboxes
            undetected_bbox_indices = list(set(range(len(bboxes_gt))).difference(det_bboxes_binary)) 
            # print(undetected_bbox_indices)
            for bbox_ind_gt in undetected_bbox_indices:
                instance_name_gt = result['all_gt']['instance_names'][bbox_ind_gt]
                conf_mat_0[instance_names.index(instance_name_gt),instance_names.index('Background')] += 1
                conf_mat_1[instance_names.index(instance_name_gt),instance_names.index('Background')] += 1


            ious_multiclass_mmrotate = rbbox_overlaps(torch.FloatTensor(bboxes_multiclass_mmrotate), torch.FloatTensor(bboxes_gt))
            det_bboxes_multiclass = []          
            for i,iou in enumerate(ious_multiclass_mmrotate):
                # ROW: detected bboxes
                # COL: gt bboxes
                if result['det_multiclass_mmrotate']['scores'][i]<confidence_score:
                    continue

                try:
                    bbox_ind_gt = np.argmax(iou)
                    det_bboxes_multiclass.append(int(bbox_ind_gt))
                    iou_score = iou[bbox_ind_gt]
                except:
                    continue

                if iou_score>iou_threshold:
                    instance_name_gt = result['all_gt']['instance_names'][bbox_ind_gt]
                else:
                    instance_name_gt = 'Background'

                instance_name_multiclass_mmrotate = result['det_multiclass_mmrotate']['instance_names'][i]
                conf_mat_2[instance_names.index(instance_name_gt),instance_names.index(instance_name_multiclass_mmrotate)] += 1
        
            ## Process not detected bboxes
            undetected_bbox_indices = list(set(range(len(bboxes_gt))).difference(det_bboxes_multiclass)) 
            for bbox_ind_gt in undetected_bbox_indices:
                instance_name_gt = result['all_gt']['instance_names'][bbox_ind_gt]
                conf_mat_2[instance_names.index(instance_name_gt),instance_names.index('Background')] += 1
            # break
        # print(conf_mat_0)
        # print(conf_mat_2)
            # print(bboxes_gt)
            # if file_name_i==5:
                # break
            # precision, recall = self.get_precision_recall(conf_mat_2)
            # precisions.append(precision)

        # print(instance_names)
        # print('BASELINE RESULTS')

        # average_precision = []
        # for i,prec in enumerate(np.arrprecisions):
            # average_precision
        # average_precision = np.array(precisions).nanmean(axis=0)
        # average_precision = np.nanmean(np.array(precisions),axis=0)
        # for i, instance_name in enumerate(instance_names):
        #     av_pre = average_precision[i]*100
        #     print(instance_name,f'{av_pre:.4f}')
        print('\nDETECTION (My approach)')
        self.get_det_accuracy(conf_mat_0)
        print('\nDETECTION (Baseline)')
        self.get_det_accuracy(conf_mat_2)

    def get_precision_recall(self,conf_matrix):

        precision = np.zeros(shape=(conf_matrix.shape[0]))
        recall = np.zeros(shape=(conf_matrix.shape[0]))
        average_precision = np.zeros(shape=(conf_matrix.shape[0]))

        for i in range(conf_matrix.shape[0]): ## Row is GT
            tp = 0
            fp = 0
            fn = 0
            for j in range(conf_matrix.shape[1]):
                # Precision
                if (i == conf_matrix.shape[0]-1) and (j != conf_matrix.shape[0]-1):
                    fn += conf_matrix[i,j]
                elif (j == conf_matrix.shape[0]-1) and (i != conf_matrix.shape[0]-1):
                    fp += conf_matrix[i,j]
                # elif (j == conf_matrix.shape[0]-1) and (i == conf_matrix.shape[0]-1):
                #     continue
                elif i == j:
                    tp = conf_matrix[i,j]
                else:
                    fn += conf_matrix[i,j]
                    fp += conf_matrix[j,i]
                # print(tp,fp,fn)
            precision[i] = tp/(tp+fp)
            recall[i] = tp/(tp+fn)
        return precision, recall
        # print(precision)
        # print(recall)
        # print(precision*recall/0.1)

    def get_det_accuracy(self,conf_matrix):

        # t_cls = 0
        # f_cls = 0
        t_det = 0
        f_det = 0

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ## Detection results
                if (i==conf_matrix.shape[0]-1) and (j==conf_matrix.shape[1]-1):
                    t_det += conf_matrix[i,j]
                elif (i==conf_matrix.shape[0]-1) or (j==conf_matrix.shape[1]-1):
                    f_det += conf_matrix[i,j]
                else:
                    t_det += conf_matrix[i,j]
        # for i in range(conf_matrix.shape[0]-1):
        #     for j in range(conf_matrix.shape[1]-1):                
        #         ### Classification results
        #         if i==j:
        #             t_cls +=conf_matrix[i,j]
        #         else:
        #             f_cls +=conf_matrix[i,j]

        # acc_cls = t_cls/(t_cls+f_cls)
        acc_det = t_det/(t_det+f_det)
        # print(f'accuracy classification: {acc_cls}')
        print(f'accuracy detection: {acc_det}')
               
# 
    def get_conf_matrices(self):
        patch_results_folder = self.settings['results']['patch_folder']
        instance_names = self.settings['original']['multiclass_instance_names'] 
        
        conf_mat_0 = np.zeros(shape=(len(instance_names)+1,len(instance_names)+1)) # BINARY MMROTATE + ORIGINAL CUTOUT
        conf_mat_1 = np.zeros(shape=(len(instance_names)+1,len(instance_names)+1)) # BINARY MMROTATE + ORTHOGONAL CUTOUT
        conf_mat_2 = np.zeros(shape=(len(instance_names)+1,len(instance_names)+1)) # MULTICLASS MMROTATE

        for file_name in os.listdir(patch_results_folder):
            patch_results_path = os.path.join(patch_results_folder,file_name)
            with open(patch_results_path,'r') as f:
                patch_result = json.load(f)

            ### MMROTATE BINARY + ORIGINAL CUTOUT
            for my_id in patch_result['det_binary_mmrotate']['ids']:
                if my_id < 0:
                    conf_mat_0[-1,-1] += 1
                    conf_mat_1[-1,-1] += 1
            for i,instance_name in enumerate(patch_result['det_binary_mmrotate_gt']['instance_names']):
                conf_mat_0[instance_names.index(instance_name),instance_names.index(patch_result['original_cutout']['instance_names'][i])] += 1
                conf_mat_1[instance_names.index(instance_name),instance_names.index(patch_result['orthogonal_cutout']['instance_names'][i])] += 1

            ### MMROTATE MULTICLASS
            count = 0
            for i,my_id in enumerate(patch_result['det_multiclass_mmrotate']['ids']):
                if my_id < 0:
                    conf_mat_2[-1,-1] += 1
                else:
                    instance_name_det = patch_result['det_multiclass_mmrotate']['instance_names'][i]
                    instance_name_gt = patch_result['det_multiclass_mmrotate_gt']['instance_names'][count]
                    conf_mat_2[instance_names.index(instance_name_gt),instance_names.index(instance_name_det)] += 1
                    count += 1
        # print(conf_mat_0)
        # print(conf_mat_1)
        # df_cm = pd.DataFrame(conf_mat_0, index =self.instance_names.append('Background'),
        #           columns = self.instance_names.append('Background'))
        # plt.figure(figsize = (10,7))
        # sn.heatmap(df_cm, annot=True)
        # plt.show()


