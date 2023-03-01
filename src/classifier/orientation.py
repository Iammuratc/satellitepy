import cv2
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose
import traceback

from src.data.cutout.tools import Tools 
from src.data.cutout.geometry import BBox 
# from src.utilities import Utilities
from src.classifier.classifier import Classifier
from src.transforms import ToTensor, Normalize, AddAxis
class ClassifierOrientation(Classifier):
    """
        Fixes the orientation of airplane cutouts
    """
    def __init__(self, settings):
        self.settings = settings
        self.cutout_tools = Tools({'tasks':['bbox','seg']})
        self.template = self.get_template()
        # self.model_utils = self.get_model_utils()
        self.exp_settings = self.get_exp_settings()
        # print(self.exp_settings)
        super(ClassifierOrientation, self).__init__(data_settings=None,exp_settings=self.exp_settings)

    # def get_model_utils(self):
    def get_exp_settings(self):
        with open(self.settings['model_config'],'r') as f:
            exp_settings = json.load(f)
        # return Utilities(model_settings)
        return exp_settings


    def get_model(self):

        model = super().get_model()
        # model = super().load_checkpoint(model,model_path=self.exp_settings['model']['path'],is_train=False)
        model.load_state_dict(torch.load(self.exp_settings['model']['path']))

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.cuda()
        return model

    def resize_data(self, img, bbox_corners):
        patch_size = self.exp_settings['patch']['size']
        img_resized = cv2.resize(
            img,
            dsize=(
                patch_size,
                patch_size),
            interpolation=cv2.INTER_LINEAR)
        
        scale = img.shape[0:2]/np.array([patch_size,patch_size])
        # print(scale)
        # new_x_coordinate = x_coordinate/(original_x_length/new_x_length)
        # new_y_coordinate = y_coordinate/(original_y_length/new_y_length)
        bbox_corners = bbox_corners / scale[::-1]
        return img_resized, bbox_corners

    def inference_model(self,model,img,transform):
        sample = {'image':img}
        sample = transform(sample)
        # print(sample['image'])
        pred = model(sample['image'].to('cuda'))
        # print(pred.shape)
        return pred.cpu().detach().numpy()[0,0,:,:] # long()


    def fix_orientation(self,generator=None,save_folder=None):
        if generator is None:
            generator = self.get_cutout_generator()
        if save_folder is None:
            save_folder = '/home/murat/Projects/airplane_recognition/data/fair1m/val/cutouts/orthogonal_images_unet'

        model = self.get_model()
        model.eval()
        transform=Compose(
                [ToTensor(), Normalize(task=None), AddAxis()])


        # break_point =9
        # fig,ax = plt.subplots(break_point,4)#,sharex=True,sharey=True)
        # copy_later = []
        for i, item in enumerate(generator):
            # if os.path.exists
            # if i != 0:
            #     continue
            # img, bbox_corners, img_ort, bbox_corners_ort = item
            img, bbox_corners, file_name = item
            img_resized, bbox_corners_resized = self.resize_data(img,bbox_corners)

            airplane_contours = self.inference_model(model,img_resized,transform)
            airplane_contours_original = cv2.resize(airplane_contours,dsize=(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)
            
            # print(airplane_contours.shape)
            my_bbox = BBox(corners=bbox_corners)
            if my_bbox.get_direction()=='clockwise':
                bbox_corners = my_bbox.switch_direction(bbox_corners)

            ### RESIZED Shift the bbox corner points
            # bbox_corners_rotated = [np.roll(np.array(bbox_corners_resized),shift=i,axis=0) for i in range(4)]
            bbox_corners_rotated = [np.roll(np.array(bbox_corners),shift=i,axis=0) for i in range(4)]

            ### RESIZED Register the template to the rotated bboxes
            # try:
            # templates = [self.register_template(img=img_resized,bbox=bbox_corners) for bbox_corners in bbox_corners_rotated]
            templates = []
            bbox_corners_templates = []
            for bbox_corners in bbox_corners_rotated:
                template, bbox_corners_template = self.register_template(img=img,bbox=bbox_corners)
                templates.append(template)
                bbox_corners_templates.append(bbox_corners_template)

            img_template_dot_prod = [np.dot(airplane_contours_original.flatten(),template.flatten())/np.count_nonzero(template.flatten()) for template in templates]
            ### Template is resized, find the not resized template
            template_match_index = np.argmax(img_template_dot_prod)
            # print(template_match_index)
            bbox_corners_matched = bbox_corners_templates[template_match_index]


            img_rotated, mask_rotated, bbox_rotated = self.cutout_tools.rotate_cutout(img=img,mask=airplane_contours_original,bbox=bbox_corners_matched,margin=20,angle='orthogonal')
            
            ### ORTHOGONAL ORIGINAL CUTOUT 
            # bbox_corners_corrected = bbox_corners_rotated[template_match_index]

            image_name = os.path.splitext(file_name)[0]
            ### SAVE IMAGES
            # try:
            # cv2.imwrite(os.path.join(orthogonal_cutout_save_folder,file_name),img_rotated)
            # cv2.imwrite(os.path.join(save_folder,file_name),img_rotated)
            # except:
            #     copy_later.append(image_name)
            #     traceback.print_exc()
            ### SAVE SAMPLES
            template_matched = templates[template_match_index]
            result_folder = os.path.join(self.settings['sample_result_folder'],image_name)
            # print(result_folder)
            os.makedirs(result_folder,exist_ok=True)
            # save rotated cutouts separately to a folder for finding the problematic ones
            result_orthogonal_folder = os.path.join(self.settings['sample_result_folder'],'orthogonals')
            os.makedirs(result_orthogonal_folder,exist_ok=True)
            cv2.imwrite(os.path.join(result_orthogonal_folder,f'{image_name}.png'),img_rotated)

            cv2.imwrite(os.path.join(result_folder,'template_matched.png'),template_matched)
            cv2.imwrite(os.path.join(result_folder,'rotated.png'),img_rotated)
            cv2.imwrite(os.path.join(result_folder,'original.png'),img)
            cv2.imwrite(os.path.join(result_folder,'unet_result.png'),airplane_contours_original*255)
            for i_t,template in enumerate(templates):
                cv2.imwrite(os.path.join(result_folder,f'template_{i_t}.png'),template)
            if i ==50:
                break
            ### PLOT TEMPLATES
            # ax[i,0].imshow(templates[0])
            # my_bbox.plot_bbox(bbox_corners_templates[0],ax=ax[i,0])
            # ax[i,1].imshow(templates[1])
            # my_bbox.plot_bbox(bbox_corners_templates[1],ax=ax[i,1])
            # ax[i,2].imshow(templates[2])
            # my_bbox.plot_bbox(bbox_corners_templates[2],ax=ax[i,2])
            # ax[i,3].imshow(templates[3])
            # PLOT IMAGES
            # my_bbox.plot_bbox(bbox_corners_templates[3],ax=ax[i,3])
            # ax[i,0].imshow(img)
            # my_bbox.plot_bbox(bbox_corners_matched,ax=ax[i,0])
            # ax[i,1].imshow(img_rotated)
            # my_bbox.plot_bbox(bbox_rotated,ax=ax[i,1])
            # ax[i,2].imshow(airplane_contours_original*255)
            # ax[i,3].imshow(templates[template_match_index])

            # if i == break_point-1:
            #     break
        # plt.show()
        # print(copy_later)
    # def plot_figures(self,ax,i,*images):

    def get_template(self):
        template_cutout_path = self.settings['template_image_path']
        template = cv2.imread(template_cutout_path,0)
        contours_img = np.zeros_like(template)
        contours, hierarchy = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, 255, 2)
        return contours_img
    def register_template(self,img,bbox):
        
        template_shape = self.template.shape
        img_shape = img.shape
        template_bbox = np.float32([
            [0,0],
            [0,template_shape[0]],
            [template_shape[1],template_shape[0]],
            [template_shape[1],0]]) # x, y       

        template_bbox_temp = np.float32([
            [[0,0]],
            [[0,template_shape[0]]],
            [[template_shape[1],template_shape[0]]],
            [[template_shape[1],0]]]) # x, y       

        # print(template_bbox)
        # homography, mask = cv2.findHomography(np.array(template_bbox), np.array(bbox), cv2.RANSAC)
        M = cv2.getPerspectiveTransform(template_bbox, np.float32(bbox))
        # print(mask)
        # print
        template_registered = cv2.warpPerspective(self.template, M, (img_shape[1], img_shape[0]))
        transformed_bbox = cv2.perspectiveTransform(template_bbox_temp, M)
        transformed_bbox = [item[0] for item in transformed_bbox]
        # template_registered = cv2.warpPerspective(self.template, homography, (img_shape[1], img_shape[0]))
        # print(transformed_bboxes)

        return template_registered, transformed_bbox

    def get_cutout_generator(self):

        file_names = os.listdir(self.settings['cutout']['image_folder'])
        # file_names.sort()
        for file_name in file_names:
            print(file_name)
            ### BBOX
            file_name_no_ext = os.path.splitext(file_name)[0]
            bbox_path = os.path.join(self.settings['cutout']['bbox_folder'],f"{file_name_no_ext}.json")
            with open(bbox_path,'r') as f:
                bbox_labels = json.load(f)
            bbox_corners = bbox_labels['original_cutout']['bbox']['corners']

            ### CUTOUT
            image_path = os.path.join(self.settings['cutout']['image_folder'],file_name)
            img = cv2.imread(image_path,1)


            ### PLOT
            # fig, ax = plt.subplots(1)
            # # ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # ax.imshow(cv2.cvtColor(img_ort,cv2.COLOR_BGR2RGB))
            # my_bbox = BBox(corners=bbox_corners)
            # # my_bbox.plot_bbox(corners=bbox_corners, ax=ax, c='b', s=15, instance_name=None)
            # my_bbox.plot_bbox(corners=bbox_ort, ax=ax, c='b', s=15, instance_name=None)
            # plt.show()
            yield img, bbox_corners, file_name#, img_ort, bbox_corners_ort