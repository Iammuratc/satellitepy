import cv2
import numpy as np
# import matplotlib.pyplot as plt

def blend_images(img_path_1,img_path_2,alpha):
    # def read_image(img_path):
    #     img = cv2.imread(img_path)
        # img_alpha = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        # return img_alpha
        # return img

    img_1 = cv2.imread(img_path_1)
    img_2 = cv2.imread(img_path_2)

    img_blend = np.zeros_like(img_1)

    img_blend = img_1[:,:,:] * alpha + img_2[:,:,:] * (1-alpha)
    return img_blend

def merge_images(img_path_1,img_path_2):
    img_1 = cv2.imread(img_path_1,0)
    img_2 = cv2.imread(img_path_2,0)

    img_merged = np.zeros(shape=(img_1.shape[0],img_1.shape[1],3))

    img_merged[:,:,1] = img_1 * 0.5
    # img_merged[:,:,0] = img_1
    img_merged[:,:,2] = img_2

    return img_merged

if __name__ == '__main__':
    import os
    import cv2
    from src.settings.utils import get_project_folder#, get_logger, create_folder
    from src.settings.orientation import SettingsOrientation


    project_folder = get_project_folder()

    my_settings = SettingsOrientation(      
        cutout_image_folder=os.path.join(project_folder,'data','fair1m','val','cutouts','images'),
        cutout_bbox_folder=os.path.join(project_folder,'data','fair1m','val','cutouts','labels'),
        model_config=os.path.join(project_folder,'experiments','exp_11','settings.json'),   
        template_image_path=os.path.join(project_folder,'data','fair1m','airliner_template.png'),
        sample_result_folder=os.path.join(project_folder,'docs','orientation'))()


    img_name = '6010_11'

    img_names = [folder_name for folder_name in os.listdir(my_settings['sample_result_folder']) if folder_name != 'orthogonals']

    for img_name in img_names:
        for i in range(4):
            # print(img_name)
            cutout_name_1 = 'unet_result'#'original'
            cutout_name_2 = f'template_{int(i)}'
            img_path_1=os.path.join(my_settings['sample_result_folder'],img_name,f'{cutout_name_1}.png')
            img_path_2=os.path.join(my_settings['sample_result_folder'],img_name,f'{cutout_name_2}.png')

            # img_blend = blend_images(
            #     alpha=0.4)

            img_merged = merge_images(img_path_1,img_path_2)
            save_img_name =  f"{cutout_name_1}_{cutout_name_2}"

            cv2.imwrite(os.path.join(my_settings['sample_result_folder'],img_name,f'{save_img_name}.png'),img_merged)


