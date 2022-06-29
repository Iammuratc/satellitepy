import shapely.geometry
import shapely.affinity
import numpy as np
from descartes import PolygonPatch
import cv2

class Rectangle:
    '''
    This class does the followings:
    - Parametrization of rectangle corner points, i.e., points >> center, height, length, rotation angle
    - Rotation calculations of bounding boxes
    
    '''
    def __init__(self,bbox):

        if bbox.shape != (4,2):
            print('bbox shape should be (4,2)')
            return 0

        self.bbox = bbox
        self.cx, self.cy, self.h, self.w, self.angle = self.get_params(self.bbox)

        self.orthogonal_bbox=self.get_orthogonal_bbox()

    @staticmethod
    def get_params(bbox):
        cx = np.sum(bbox[:,0])/4.0
        cy = np.sum(bbox[:,1])/4.0


        x_dif = lambda i_0,i_1: bbox[i_0,0] - bbox[i_1,0]
        y_dif = lambda i_0,i_1: bbox[i_0,1] - bbox[i_1,1]

        h = np.sqrt(x_dif(0,1)**2 + y_dif(0,1)**2 )
        w = np.sqrt(x_dif(1,2)**2 + y_dif(1,2)**2 )       

        angle =np.arctan(y_dif(0,1)/x_dif(0,1))
        return [cx,cy,h,w,angle]


    def get_atan2(self):#,img_shape):
        '''
        Get the rotation matrix to rotate the airplane upwards 
        '''
        # y_max, x_max, _ = img_shape
        # print(np.rad2deg(self.angle))
        if (self.bbox[0,1]-self.bbox[1,1]>=0) and (self.bbox[0,0]-self.bbox[1,0]>=0):
            # print('q1')
            angle = 3*np.pi/2+self.angle
        elif (self.bbox[0,1]-self.bbox[1,1]>=0) and (self.bbox[0,0]-self.bbox[1,0]<=0):
            # print('q2')
            angle = np.pi/2+self.angle
        elif (self.bbox[0,1]-self.bbox[1,1]<=0) and (self.bbox[0,0]-self.bbox[1,0]<=0): 
            # print('q3')
            angle = np.pi/2+self.angle
        elif (self.bbox[0,1]-self.bbox[1,1]<=0) and (self.bbox[0,0]-self.bbox[1,0]>=0): 
            # print('q4')
            angle=3*np.pi/2+self.angle
        # M = cv2.getRotationMatrix2D((y_max/2, x_max/2), np.rad2deg(angle), 1.0)

        # return M,angle
        return angle

    def get_orthogonal_bbox(self):

        angle = -self.get_atan2()
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d((self.cx,self.cy))
        # bbox = np.atleast_2d(self.bbox)
        return np.squeeze((R @ (self.bbox.T-o.T) + o.T).T)

    @staticmethod
    def plot_bbox(bbox,ax,c):
        for i, coord in enumerate(bbox):
            # PLOT BBOX
            ax.plot([bbox[i-1][0],coord[0]],[bbox[i-1][1],coord[1]],c=c)
        ax.scatter(bbox[0][0],bbox[0][1],c='r',s=15)
        return ax

    @staticmethod
    def get_bbox_limits(bbox):
        x_min = np.amin(bbox[:,0])
        x_max = np.amax(bbox[:,0])
        y_min = np.amin(bbox[:,1])
        y_max = np.amax(bbox[:,1])
        return np.array([x_min,x_max,y_min,y_max]).astype(int)


if __name__ == "__main__":
    import numpy as np
    import shapely.wkt

    import matplotlib.pyplot as plt    
    import json
    import os
    import random
    import cv2

    # bbox = np.array([[43.0, 46.0], [47.0, 87.0], [86.0, 83.0], [81.0, 41.0]])

    # rect = Rectangle(bbox=bbox)
    # orth_bbox = rect.get_orthogonal_bbox()
    # print(rect.bbox)

    # print(orth_bbox)


    # fig,ax = plt.subplots(1)
    # ax.set_ylim([0,256])
    # ax.set_xlim([0,256])

    # rect.plot_bbox(ax=ax,bbox=rect.bbox,c='y')
   
    # rect.plot_bbox(ax=ax,bbox=orth_bbox,c='b')

    # plt.show()

    # P = np.asarray(shapely.wkt.loads('POLYGON ((51.0 3.0, 51.3 3.61, 51.3 3.0, 51.0 3.0))'))
    P = shapely.wkt.loads('POLYGON ((51.0 3.0, 51.3 3.61, 51.3 3.0, 51.6 4.0, 51.0 3.0))')
    # print(np.array(P.exterior.coords))
    bbox = np.array(P.exterior.coords)[0:4,:]
    print(bbox)
