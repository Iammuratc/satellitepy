import os
import math
import numpy as np





def padding():
    a = [[1, 2], [3, 4]]

    b = np.pad(a, ((1, 2), (2, 3)), 'constant',constant_values=0)
    
    print(b)

### REPLACE 

# Read in the file

def replace_text(file_path,old_text,new_text):
    with open(file_path, 'r') as file :
      filedata = file.read()

    # Replace the target string
    filedata = filedata.replace(old_text,new_text)

    # Write the file out again
    with open(file_path, 'w') as file:
      file.write(filedata)

def get_file_name(file):
    file_name = file.split('.')[0]
    return file_name



if __name__ == '__main__':
    # padding()

    # from PIL import Image, ExifTags
    # img = Image.open("/home/murat/Projects/airplane_detection/DATA/Gaofen/train/images/4.tif")
    # exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    # print(exif)
        # my_folder = "../DATA/Gaofen/val/label_xml"

    # for file in os.listdir(my_folder):
    #   file_path = f"{my_folder}/{file}"

    #   file_name = get_file_name(file)

    #   old_text = f"A:{chr(92)}Datasets{chr(92)}GAOFEN{chr(92)}val{chr(92)}label_xml{chr(92)}{str(int(file_name)+1000)}.xml"
    #   new_text = f"{file_name}.xml"
    #   replace_text(file_path,old_text,new_text)
    # import torch
    # unfold_mat()

    ### ROTATE BOX 
    # import matplotlib.pyplot as plt

    # import shapely.geometry
    # import numpy as np
    # # from descartes import PolygonPatch
    # c = shapely.geometry.box(-20, -10, 20, 10)
    # c = shapely.affinity.rotate(c, -0.14,use_radians=True)
    # rotated_box = shapely.affinity.translate(c, 0, 0)

    
    # # CALCULATE ANGLE FROM CORNERS
    # x,y = rotated_box.exterior.coords.xy
    # print(x)
    # print(y)
    # x_dif = x[3] - x[4]
    # y_dif = y[3] - y[4]

    # angle = np.arctan(y_dif/x_dif)
    # print("{:1.2}".format(angle))


    # fig,ax = plt.subplots(1)
    # ax = plt.gca()
    # ax.set_xlim([-30, +30])
    # ax.set_ylim([-30, +30])
    # plt.plot(*rotated_box.exterior.xy)
    # # ax.add_patch(PolygonPatch(rotated_box, fc='#04d648',alpha=0.5))

    # plt.show()

### OLD GEOMETRY
    # def plot_rotated_box(corners,ax):
    #     # for coords in corners:
    #         # print(coords)
    #     for i, coord in enumerate(corners):
    #         # PLOT BBOX
    #         ax.plot([corners[i-1][0],coord[0]],[corners[i-1][1],coord[1]],c='r')


    # labels_folder = "/home/murat/Projects/airplane_detection/DATA/Gaofen/train/patches_512/labels_original"
    # img_folder = "/home/murat/Projects/airplane_detection/DATA/Gaofen/train/patches_512/images"
    
    # # my_files = os.listdir(data_folder)
    # # i = random.randint(0, len(my_files))
    # # print(my_files[i])
    # # my_file = os.path.splitext(my_files[i])
    # my_file = '1_x_412_y_412'
    # my_dict = json.load(open(f"{labels_folder}/{my_file}.json",'r'))
    # rotated_bboxes = my_dict['rotated_bboxes']
    # img = cv2.imread(f"{img_folder}/{my_file}.png")




    # fig, ax = plt.subplots(1)
    # # ax.imshow()
    # ax = plt.gca()
    # ax.set_xlim([0, 512])
    # ax.set_ylim([0, 512])
    # for i,corners in enumerate(rotated_bboxes):
    #     # print(corners)
    #     instance_name = my_dict["instance_names"][i]
    #     rotated_rect = RotatedRect(corners=np.array(corners),parametized=False)
    #     print(f"Type: {instance_name} height: {rotated_rect.h} width: {rotated_rect.w}")

    #     rotated_rect.plot_contours(ax)
    #     plot_rotated_box(corners,ax)
    # plt.show()


    # def get_airplane_size(self,patch_size=128):

    #     my_patch_folder = self.get_patch_folder(patch_size)
    #     label_patch_folder = f"{my_patch_folder}/labels"

    #     size_dict = {}
    #     for json_file in os.listdir(label_patch_folder):
    #        patch_dict = json.load(open(f"{label_patch_folder}/{json_file}",'r'))
    #        h,w = patch_dict['size_hw']
    #        instance_name=patch_dict['instance_name']
           
    #        if instance_name not in size_dict.keys():
    #             size_dict[instance_name] = {'w':[],'h':[]}
    #        else:
    #             size_dict[instance_name]['w'].append(w)
    #             size_dict[instance_name]['h'].append(h)
    #     # print(size_dict)

    #     for instance_name in size_dict.keys():
    #         total_no = len(size_dict[instance_name]['h'])
    #         fig,ax=plt.subplots(2)
    #         fig.suptitle(f'Instance:{instance_name}, total no: {total_no}')
    #         ax[0].set_title('Height')
    #         ax[1].set_title('Width')
    #         ax[0].hist(size_dict[instance_name]['h'],bins=50)
    #         ax[1].hist(size_dict[instance_name]['w'],bins=50)
    #         plt.show()
            # plt.savefig(f"{my_patch_folder}/figures/hist_{instance_name}.png", bbox_inches='tight')