import imgaug as ia
import imgaug.augmenters as iaa
import torch
# from geometry import Transformations
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
import numpy as np

# TODO: Modify ToTensor output for other values

class Augmentations:
    def __init__(self):
        self.transformations=TransformationsUtils()
    def __call__(self,sample):
        transform_prob = 0.7
        # APPLY AUGMENTATIONS AND ADJUST BBOXES ACCORDINGLY
        sample_aug = sample.copy()

        seq = iaa.Sequential([iaa.Sometimes(transform_prob,
            iaa.Multiply((0.5, 1.5)), # change brightness
            iaa.Affine(
                rotate=(-10,10),
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                shear=(-16, 16),
                ) 
            )
        ])

        if sample['airplane_exist']: # If there is any airplane in the image, augment bboxes, else, augment image only            
            ### BOUNDING BOX OPERATIONS
            sample_aug = self.transformations.get_bbox_aug(sample_aug,seq)
            # print(sample_aug['orthogonal_bboxes'])
            # print(sample_aug['rotated_bboxes'])
        else:
            image_aug = seq(image=sample_aug['image'])
            sample_aug['image']=image_aug

        ### SHOW IMAGE
        # show_sample(sample_aug)

        # for corners in bbox_aug:
        #     print(corners)       
        #     my_rect = RotatedRect(parametized=False,corners=corners)
        #     print(my_rect.angle)
        #     ax = my_rect.plot_contours(ax)
        return sample_aug

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        AND NORMALIZE
    """

    def __call__(self, sample):
        image = sample['image']/255.0
        bbox = sample['orthogonal_bboxes']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        # print(torch.from_numpy(image).unsqueeze(dim=0).shape)
        return {'image': torch.from_numpy(image),#.unsqueeze(dim=0),
                'orthogonal_bboxes': torch.FloatTensor(bbox)}#.unsqueeze(dim=0)}#torch.from_numpy(bbox)}


class TransformationsUtils:
    def get_bbox_aug(self,sample_aug,seq):
        image=sample_aug['image']
        
        ### PRE ORTHOGONAL BOXES ########################
        # bboxes_params = sample_aug['orthogonal_bboxes']
        # my_orth_boxes = []
        # for params in bboxes_params:
        #     xc,yc,w,h = params
        #     my_orth_boxes.append(BoundingBox(x1=xc-w/2.0, x2=xc+w/2.0, y1=yc-h/2.0, y2=yc+h/2.0))
        # bboxes = BoundingBoxesOnImage(my_orth_boxes, shape=image.shape)

        ##### PRE ROTATED BOXES ######################
        rotated_bboxes = sample_aug['rotated_bboxes']
        kps = KeypointsOnImage([Keypoint(x=coord[0], y=coord[1]) for coords in rotated_bboxes for coord in coords], shape=image.shape)

        image_aug, kps_aug = seq(image=image, keypoints=kps) #bounding_boxes=bboxes,
        
        ##### POST ROTATED BOXES ######################
        rot_bboxes = np.array([keypoint.xy for keypoint in kps_aug]).reshape(-1,4,2) # eg: 3,4,2
        ### Remove airplanes outside of the image
        rot_bboxes = self.remove_out_of_image(rot_bboxes,sample_aug['patch_size'])
    
        ##### POST ORTHOGONAL BOXES ######################
        # orth_boxes = []
        # for bbox in bboxes_aug.remove_out_of_image().bounding_boxes: #.clip_out_of_image()
        #     x1,x2,y1,y2 = bbox.x1,bbox.x2,bbox.y1,bbox.y2
        #     xc,yc,w,h = (x1+x2)/2.0,(y1+y2)/2.0,np.abs(x2-x1),np.abs(y2-y1)   
        #     orth_boxes.append([xc,yc,w,h])
        orth_boxes = self.get_orthogonal_from_rotated_box(rot_bboxes)
        

        sample_aug['orthogonal_bboxes'] = orth_boxes
        sample_aug['rotated_bboxes']= rot_bboxes
        ###########################
        sample_aug['image'] = image_aug
        return sample_aug

    def remove_out_of_image(self,rotated_boxes,patch_size):
        new_rotated_boxes = []
        box_corner_threshold = 2
        for rotated_box in rotated_boxes:
            box_corner_in_patch=0
            for coord in rotated_box:
                x_coord = coord[0]
                y_coord = coord[1]
                if (0<=x_coord<=patch_size) and (0<=coord[1]<=patch_size):
                    box_corner_in_patch += 1
            if box_corner_in_patch>=box_corner_threshold:
                new_rotated_boxes.append(rotated_box)
        return new_rotated_boxes

    def get_orthogonal_from_rotated_box(self,rotated_boxes):
        orth_boxes = []
        for rotated_box in rotated_boxes:
            x_coords = rotated_box[:,0]
            y_coords = rotated_box[:,1]
            x_coord_min, x_coord_max = np.amin(x_coords),np.amax(x_coords)
            y_coord_min, y_coord_max = np.amin(y_coords),np.amax(y_coords)
            h = y_coord_max - y_coord_min
            w = x_coord_max - x_coord_min
            orth_boxes.append([x_coord_min+w/2.0, y_coord_min+h/2.0, w, h])
        return orth_boxes 