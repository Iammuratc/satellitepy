import shapely.geometry
import shapely.affinity
import numpy as np
from descartes import PolygonPatch

# TODO: get_params is slightly off
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
                print('Corners shape should (4,2)')
                return 0

            self.corners = corners
            self.params = self.get_params()
    def get_params(self):
        self.cx = np.sum(self.corners[:,0])/4.0
        self.cy = np.sum(self.corners[:,1])/4.0



        i_0 = np.argmin(self.corners[:,0],axis=None)
        print(i_0)
        i_1 = row_no+1 if row_no+1<len(self.corners) else 0
        print(i_1)

        # x_dif = self.corners[0,0] - self.corners[1,0]
        # y_dif = self.corners[0,1] - self.corners[1,1]
        y_dif = lambda i_0,i_1: self.corners[i_0,1] - self.corners[i_1,1]
        x_dif = lambda i_0,i_1: self.corners[i_0,0] - self.corners[i_1,0]

        # print(x_dif,y_dif)
        self.w = np.sqrt( x_dif(i_0,i_1)**2 + (y_dif(i_0,i_1))**2 )
        self.h = np.sqrt( (x_dif(i_0-1))**2 + (self.corners[0,1] - self.corners[3,1])**2 )

        # angle = np.arctan(y_dif/x_dif)
        # print('corners,', self.corners)
        angle = np.arctan(x_dif/y_dif)
        # if angle < 0:

            # self.angle = -(np.pi-np.arctan(y_dif/x_dif))
            # self.angle = np.arctan(y_dif/x_dif)
        # else:
        self.angle= angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle,use_radians=True)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

    def plot_contours(self,ax,fc='#04d648'):
        ax.add_patch(PolygonPatch(self.get_contour(), fc=fc, alpha=0.5))
        return ax
