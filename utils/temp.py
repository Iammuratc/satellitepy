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