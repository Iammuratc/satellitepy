import cv2
import torch
import numpy as np
import math

from satellitepy.dataset.bbavector.draw_gaussian import draw_umich_gaussian, gaussian_radius
from satellitepy.dataset.bbavector.transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
from satellitepy.dataset.bbavector import data_augment
from satellitepy.data.utils import get_satellitepy_dict_values, get_task_dict #, merge_satellitepy_task_values

class Utils:
    def __init__(self, tasks, input_h=None, input_w=None, down_ratio=None, K=1000):
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.tasks = tasks
        self.max_objs = 1000
        self.image_distort =  data_augment.PhotometricDistort()
        

    def data_transform(self, image, annotation, augmentation):
        # only do random_flip augmentation to original images
        crop_size = None
        crop_center = None

        boxes = []
        masks = []
        if "masks" in annotation:
            masks.append(annotation["masks"])
        if "hbboxes" in annotation:
            boxes.append(annotation["hbboxes"])
        if "obboxes" in annotation:
            boxes.append(annotation["obboxes"])

        if augmentation:
            crop_size, crop_center = random_crop_info(h=image.shape[0], w=image.shape[1])
            image, masks, boxes, crop_center = random_flip(image, masks, boxes, crop_center)
        if crop_center is None:
            crop_center = np.asarray([float(image.shape[1])/2, float(image.shape[0])/2], dtype=np.float32)
        if crop_size is None:
            crop_size = [max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0])]  # init
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=augmentation)
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        for idx, m in enumerate(masks):
            if m is not None:
                masks[idx] = cv2.warpAffine(
                    src=m, 
                    M=M, 
                    dsize=(self.input_w, self.input_h), 
                    flags=cv2.INTER_LINEAR
                )
        for idx, box in enumerate(boxes):
            for idx_1, b in enumerate(box):
                if b is not None:
                    new_box = np.concatenate([b, np.ones((b.shape[0], 1))], axis=1)
                    new_box = np.matmul(new_box, np.transpose(M))
                    boxes[idx][idx_1] = np.asarray(new_box, np.float32)

        out_annotations = {}
        check_boxes = []
        if "obboxes" in annotation:
            check_boxes = boxes[-1]
        if "hbboxes" in annotation:
            if len(check_boxes) > 0:
                for idx, (e_b, b) in enumerate(zip(check_boxes, boxes[0])):
                    if e_b is None:
                        check_boxes[idx] = b
            else:
                check_boxes = boxes[0]

        if len(check_boxes) > 0:
            size_thresh = 3
            out_hbb = []
            out_obb = []
            for idx, pt_old in enumerate(check_boxes):
                if (pt_old<0).any() or (pt_old[:,0]>self.input_w-1).any() or (pt_old[:,1]>self.input_h-1).any():
                    pt_new = np.float32(pt_old).copy()
                    pt_new[:,0] = np.minimum(np.maximum(pt_new[:,0], 0.), self.input_w - 1)
                    pt_new[:,1] = np.minimum(np.maximum(pt_new[:,1], 0.), self.input_h - 1)
                    iou = ex_box_jaccard(pt_old.copy(), pt_new.copy())
                    if iou>0.6:
                        rect = cv2.minAreaRect(pt_new/self.down_ratio)
                        width, height = rect[1][0], rect[1][1]
                        if width>size_thresh and height>size_thresh:
                            if "hbboxes" in annotation:
                                if annotation["hbboxes"][idx] is not None:
                                    out_hbb.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1]])
                                else: 
                                    out_hbb.append(None)
                            if "obboxes" in annotation:
                                if annotation["obboxes"][idx] is not None:
                                    out_obb.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                                else:
                                    out_obb.append(None)
                            for k in annotation.keys():
                                if k != "hbboxes" and k != "obboxes":
                                    out_annotations.setdefault(k, [])
                                    out_annotations[k].append(annotation[k][idx])
                else:
                    rect = cv2.minAreaRect(np.float32(pt_old)/self.down_ratio)
                    width, height = rect[1][0], rect[1][1]
                    if width>size_thresh and height>size_thresh:
                        if "hbboxes" in annotation:
                            if annotation["hbboxes"][idx] is not None:
                                out_hbb.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1]])
                            else: 
                                out_hbb.append(None)
                        if "obboxes" in annotation:
                            if annotation["obboxes"][idx] is not None:
                                out_obb.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                            else:
                                out_obb.append(None)
                        for k in annotation.keys():
                            if k != "hbboxes" and k != "obboxes":
                                out_annotations.setdefault(k, [])
                                out_annotations[k].append(annotation[k][idx])

        if "hbboxes" in annotation and len(out_hbb) > 0:
            out_annotations["hbboxes"] = np.asarray(out_hbb, np.float32)
        if "obboxes" in annotation and len(out_obb) > 0:
            out_annotations["obboxes"] = np.asarray(out_obb, np.float32)

        for k in out_annotations.keys():
            if k != "hbboxes" and k != "obboxes":
                out_annotations[k] = np.asarray(out_annotations[k])

        return image, out_annotations


    # def __len__(self):
    #     return len(self.img_ids)

    # def processing_test(self, image, input_h, input_w):
    #     image = cv2.resize(image, (input_w, input_h))
    #     out_image = image.astype(np.float32) / 255.
    #     out_image = out_image - 0.5
    #     out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    #     out_image = torch.from_numpy(out_image)
    #     return out_image

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1


    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new


    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))
        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        ret = {
            "input": image
        }
        for k in annotation.keys():
            if k == "masks":
                ret[k] = annotation[k]
            if k not in ["obboxes", "hbboxes", "masks", "cls_coarse-class"]:
                # todo: we probably have to define 0 as background class / non-object class
                ret[k] = np.zeros((self.max_objs), dtype=np.float32)
                for idx, v in enumerate(annotation[k]):
                    ret[k][idx] = v

        num_classes = len(get_task_dict("coarse-class"))
        ret["cls_coarse-class"] = np.zeros((num_classes, image_h, image_w), dtype=np.float32)

        if "obboxes" in annotation.keys():
            wh = np.zeros((self.max_objs, 10), dtype=np.float32)
            cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            ind = np.zeros((self.max_objs), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
            num_objs = min(annotation['obboxes'].shape[0], self.max_objs)
            for k in range(num_objs):
                if isinstance(annotation["obboxes"][k], np.float32):
                    continue
                rect = annotation['obboxes'][k, :]
                cen_x, cen_y, bbox_w, bbox_h, theta = rect
                radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                ct = np.asarray([cen_x, cen_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(ret["cls_coarse-class"][annotation['cls_coarse-class'][k]], ct_int, radius)
                ind[k] = ct_int[1] * image_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

                bl = pts_4[0,:]
                tl = pts_4[1,:]
                tr = pts_4[2,:]
                br = pts_4[3,:]

                tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
                rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
                bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
                ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

                if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                    tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
                wh[k, 0:2] = tt - ct
                wh[k, 2:4] = rr - ct
                wh[k, 4:6] = bb - ct
                wh[k, 6:8] = ll - ct
                w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
                wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
                jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
                if jaccard_score<0.95:
                    cls_theta[k, 0] = 1
            ret["obboxes_params"] = wh
            ret["obboxes_offset"] = reg
            ret["obboxes_theta"] = cls_theta
            ret["ind"] = ind
            ret["reg_mask"] = reg_mask

        if "hbboxes" in annotation.keys():
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            ind = np.zeros((self.max_objs), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
            num_objs = min(annotation['obboxes'].shape[0], self.max_objs)
            for k in range(num_objs):
                if isinstance(annotation["hbboxes"][k], np.float32):
                    continue
                rect = annotation['hbboxes'][k, :]
                cen_x, cen_y, bbox_w, bbox_h = rect
                radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                radius = max(0, int(radius))
                ct = np.asarray([cen_x, cen_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(ret["cls_coarse-class"][annotation['cls_coarse-class'][k]], ct_int, radius)
                ind[k] = ct_int[1] * image_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                pts_4 = np.array([
                    [cen_x - bbox_w / 2, cen_y + bbox_h / 2],
                    [cen_x - bbox_w / 2, cen_y - bbox_h / 2],
                    [cen_x + bbox_w / 2, cen_y + bbox_h / 2],
                    [cen_x + bbox_w / 2, cen_y + bbox_h / 2],
                ])
                w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
                wh[k] = 1. * w_hbbox, 1. * h_hbbox

            ret["hbboxes_params"] = wh
            ret["hbboxes_offset"] = reg
            if "ind" not in ret:
                ret["ind"] = ind
            if "reg_mask" not in ret:
                ret["reg_mask"] = reg_mask

        for k, v in ret.items():
            ret[k] = torch.from_numpy(v)

        return ret

    # def __getitem__(self, index):
    #     image = self.load_image(index)
    #     image_h, image_w, c = image.shape
    #     if self.phase == 'test':
    #         img_id = self.img_ids[index]
    #         image = self.processing_test(image, self.input_h, self.input_w)
    #         return {'image': image,
    #                 'img_id': img_id,
    #                 'image_w': image_w,
    #                 'image_h': image_h}

    #     elif self.phase == 'train':
    #         annotation = self.load_annotation(index)
    #         image, annotation = self.data_transform(image, annotation)
    #         data_dict = self.generate_ground_truth(image, annotation)
    #         return data_dict


    # def load_img_ids(self):
    #     """
    #     Definition: generate self.img_ids
    #     Usage: index the image properties (e.g. image name) for training, testing and evaluation
    #     Format: self.img_ids = [list]
    #     Return: self.img_ids
    #     """
    #     return None

    # def load_image(self, index):
    #     """
    #     Definition: read images online
    #     Input: index, the index of the image in self.img_ids
    #     Return: image with H x W x 3 format
    #     """
    #     return None

    # def load_annoFolder(self, img_id):
    #     """
    #     Return: the path of annotation
    #     Note: You may not need this function
    #     """
    #     return None

    # def load_annotation(self, index):
    #     """
    #     Return: dictionary of {'pts': float np array of [bl, tl, tr, br], 
    #                             'cat': int np array of class_index}
    #     Explaination:
    #             bl: bottom left point of the bounding box, format [x, y]
    #             tl: top left point of the bounding box, format [x, y]
    #             tr: top right point of the bounding box, format [x, y]
    #             br: bottom right point of the bounding box, format [x, y]
    #             class_index: the category index in self.category
    #                 example: self.category = ['ship]
    #                          class_index of ship = 0
    #     """
    #     return None

    # def dec_evaluation(self, result_path):
    #     return None

