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
    def __init__(self,bbox):#**kwargs):
        # if parametized == True:
        #     self.cx = kwargs['cx']
        #     self.cy = kwargs['cy']
        #     self.w = kwargs['w']
        #     self.h = kwargs['h']
        #     self.angle = kwargs['angle']
        # else:
        """
            bbox = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
            x0,y0 = top left of the airplane bbox
            x1,y1 = bottom left
            x2,y2 = bottom right
            x3,y3 = top right
        """
        if bbox.shape != (4,2):
            print('bbox shape should be (4,2)')
            return 0

        self.bbox = bbox
        self.params = self.get_params()

    def get_params(self):
        self.cx = np.sum(self.bbox[:,0])/4.0
        self.cy = np.sum(self.bbox[:,1])/4.0


        x_dif = lambda i_0,i_1: self.bbox[i_0,0] - self.bbox[i_1,0]
        y_dif = lambda i_0,i_1: self.bbox[i_0,1] - self.bbox[i_1,1]

        self.h = np.sqrt(x_dif(0,1)**2 + y_dif(0,1)**2 )
        self.w = np.sqrt(x_dif(1,2)**2 + y_dif(1,2)**2 )       

        self.angle =np.arctan(y_dif(0,1)/x_dif(0,1))

    
    # def get_contour(self,rotate=True):

    #     cx,cy,h,w,angle = self.cx,self.cy,self.h,self.w,self.angle
    #     c = shapely.geometry.box(-h/2.0, -w/2.0, h/2.0, w/2.0)
    #     # if rotate:
    #     rc = shapely.affinity.rotate(c, angle,use_radians=True)
    #     # else:
    #     #     rc = shapely.affinity.rotate(c, np.pi/2,use_radians=True)

    #     rc = shapely.affinity.translate(rc, cx, cy)
    #     print(rc)
    #     return rc
# 
    # def plot_contours(self,ax,rotate,fc='#04d648'):

        # if 'params' in kwargs.keys():
        #     params = kwargs['params'] # center_x,center_y,height,width,rotation_angle
        # else:
        # params = 
        # ax.add_patch(PolygonPatch(self.get_contour(rotate),  ec=fc,fill=False)) #alpha=0.5 fc=fc,
        # ax.scatter(self.cx+self.h/2,self.cy+self.w/2,c='r',s=15)
        # return ax

    # def plot_corners(self,ax):
    #     for i, coord in enumerate(self.bbox):
    #         # PLOT BBOX
    #         ax.plot([self.bbox[i-1][0],coord[0]],[self.bbox[i-1][1],coord[1]],c='y')
    #     ax.scatter(self.bbox[0][0],self.bbox[0][1],c='r',s=15)
    #     return ax

    def plot_bbox(self,bbox,ax,c):
        for i, coord in enumerate(bbox):
            # PLOT BBOX
            ax.plot([bbox[i-1][0],coord[0]],[bbox[i-1][1],coord[1]],c=c)
        ax.scatter(bbox[0][0],bbox[0][1],c='r',s=15)
        return ax

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
        print(angle)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d((self.cx,self.cy))
        # bbox = np.atleast_2d(self.bbox)
        # print(bbox.shape)
        # print(o.shape)
        return np.squeeze((R @ (self.bbox.T-o.T) + o.T).T)


if __name__ == "__main__":
    import numpy as np
    import shapely.geometry

    import matplotlib.pyplot as plt    
    import json
    import os
    import random
    import cv2

    bbox = np.array([[43.0, 46.0], [47.0, 87.0], [86.0, 83.0], [81.0, 41.0]])

    rect = Rectangle(bbox=bbox)
    orth_bbox = rect.get_orthogonal_bbox()
    print(rect.bbox)

    print(orth_bbox)


    fig,ax = plt.subplots(1)
    ax.set_ylim([0,256])
    ax.set_xlim([0,256])

    rect.plot_bbox(ax=ax,bbox=rect.bbox,c='y')
   
    rect.plot_bbox(ax=ax,bbox=orth_bbox,c='b')

    plt.show()
