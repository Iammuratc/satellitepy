import shapely.geometry
import shapely.affinity
import numpy as np
from descartes import PolygonPatch

# TODO: get_params is off

class RotatedRect:
    def __init__(self, parametized,**kwargs):
        if parametized == True:
            self.cx = kwargs['cx']
            self.cy = kwargs['cy']
            self.w = kwargs['w']
            self.h = kwargs['h']
            self.angle = kwargs['angle']
        else:
            """
                corners = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
                x0,y0 = bottom left
                x1,y1 = bottom right
                x2,y2 = top right
                x3,y3 = top left
            """
            corners = kwargs['corners']
            if corners.shape != (4,2):
                print('Corners shape should be (4,2)')
                return 0

            self.corners = corners
            self.params = self.get_params()
    def get_params(self):
        self.cx = np.sum(self.corners[:,0])/4.0
        self.cy = np.sum(self.corners[:,1])/4.0


        x_dif = lambda i_0,i_1: self.corners[i_0,0] - self.corners[i_1,0]
        y_dif = lambda i_0,i_1: self.corners[i_0,1] - self.corners[i_1,1]

        self.h = np.sqrt(x_dif(0,1)**2 + y_dif(0,1)**2 )
        self.w = np.sqrt(x_dif(1,2)**2 + y_dif(1,2)**2 )       

        self.angle = np.arctan(y_dif(0,1)/x_dif(0,1))


    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle,use_radians=True)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def plot_contours(self,ax,fc='#04d648'):
        ax.add_patch(PolygonPatch(self.get_contour(), fc=fc, alpha=0.5))
        return ax





if __name__ == "__main__":
    import numpy as np
    import shapely.geometry

    import matplotlib.pyplot as plt    
    import json
    import os
    import random
    import cv2

    def plot_rotated_box(corners,ax):
        # for coords in corners:
            # print(coords)
        for i, coord in enumerate(corners):
            # PLOT BBOX
            ax.plot([corners[i-1][0],coord[0]],[corners[i-1][1],coord[1]],c='r')


    labels_folder = "/home/murat/Projects/airplane_detection/DATA/Gaofen/train/patches_512/labels_original"
    img_folder = "/home/murat/Projects/airplane_detection/DATA/Gaofen/train/patches_512/images"
    
    # my_files = os.listdir(data_folder)
    # i = random.randint(0, len(my_files))
    # print(my_files[i])
    # my_file = os.path.splitext(my_files[i])
    my_file = '1_x_412_y_412'
    my_dict = json.load(open(f"{labels_folder}/{my_file}.json",'r'))
    rotated_bboxes = my_dict['rotated_bboxes']
    img = cv2.imread(f"{img_folder}/{my_file}.png")




    fig, ax = plt.subplots(1)
    # ax.imshow()
    ax = plt.gca()
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    for i,corners in enumerate(rotated_bboxes):
        # print(corners)
        instance_name = my_dict["instance_names"][i]
        rotated_rect = RotatedRect(corners=np.array(corners),parametized=False)
        print(f"Type: {instance_name} height: {rotated_rect.h} width: {rotated_rect.w}")

        rotated_rect.plot_contours(ax)
        plot_rotated_box(corners,ax)
    plt.show()