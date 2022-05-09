import shapely.geometry
import shapely.affinity
import numpy as np
from descartes import PolygonPatch
import cv2

class Rectangle:
    '''
    This class enables the followings:
    - Parametrization of rectangle corner points, i.e., points >> center, height, length, rotation angle
    - Rotation calculations of bounding boxes
    
    '''
    def __init__(self,corners):#**kwargs):
        # if parametized == True:
        #     self.cx = kwargs['cx']
        #     self.cy = kwargs['cy']
        #     self.w = kwargs['w']
        #     self.h = kwargs['h']
        #     self.angle = kwargs['angle']
        # else:
        """
            corners = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
            x0,y0 = top left of the airplane bbox
            x1,y1 = bottom left
            x2,y2 = bottom right
            x3,y3 = top right
        """
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

        self.angle =np.arctan(y_dif(0,1)/x_dif(0,1))


    def get_contour(self,rotate=True):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-h/2.0, -w/2.0, h/2.0, w/2.0)
        if rotate:
            rc = shapely.affinity.rotate(c, self.angle,use_radians=True)
        else:
            rc = shapely.affinity.rotate(c, np.pi/2,use_radians=True)

        return shapely.affinity.translate(rc, self.cx, self.cy)

    def plot_contours(self,ax,rotate,fc='#04d648'):
        ax.add_patch(PolygonPatch(self.get_contour(rotate), fc=fc, alpha=0.5))
        return ax


    def get_rotation_matrix_upwards(self,img_shape):
        '''
        Get the rotation matrix to rotate the airplane upwards 
        '''
        y_max, x_max, _ = img_shape
        if (self.corners[0,1]-self.corners[1,1]>=0) and (self.corners[0,0]-self.corners[1,0]>=0): 
            # q1
            M = cv2.getRotationMatrix2D((y_max/2, x_max/2), np.rad2deg(3*np.pi/2+self.angle), 1.0)
        elif (self.corners[0,1]-self.corners[1,1]>=0) and (self.corners[0,0]-self.corners[1,0]<=0):
            # q2
            M = cv2.getRotationMatrix2D((y_max/2, x_max/2), np.rad2deg(np.pi/2+self.angle), 1.0)                
        elif (self.corners[0,1]-self.corners[1,1]<=0) and (self.corners[0,0]-self.corners[1,0]<=0): 
            # q3
            M = cv2.getRotationMatrix2D((y_max/2, x_max/2), np.rad2deg(np.pi/2+self.angle), 1.0)                
        elif (self.corners[0,1]-self.corners[1,1]<=0) and (self.corners[0,0]-self.corners[1,0]>=0): 
            # q4
            M = cv2.getRotationMatrix2D((y_max/2, x_max/2), np.rad2deg(3*np.pi/2+self.angle), 1.0)

        return M



if __name__ == "__main__":
    import numpy as np
    import shapely.geometry

    import matplotlib.pyplot as plt    
    import json
    import os
    import random
    import cv2

    corners = np.array([[43.0, 46.0], [47.0, 87.0], [86.0, 83.0], [81.0, 41.0]])

    rotated_rect = RotatedRect(parametized=False,corners=corners)
    print(rotated_rect.angle)

    fig,ax = plt.subplots(1)
    ax.set_ylim([0,256])
    ax.set_xlim([0,256])

    rotated_rect.plot_contours(ax)
    plt.show()
