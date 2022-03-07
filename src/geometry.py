import shapely.geometry
import shapely.affinity
import numpy as np
from descartes import PolygonPatch


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
        self.cx = np.sum(self.corners[:,0])/4
        self.cy = np.sum(self.corners[:,1])/4
        self.h = np.sqrt( (self.corners[0,0] - self.corners[3,0])**2 + (self.corners[0,1] - self.corners[3,1])**2 )

        x_dif = self.corners[0,0] - self.corners[1,0]
        y_dif = self.corners[0,1] - self.corners[1,1]
        # print(x_dif,y_dif)
        self.w = np.sqrt( (x_dif)**2 + (y_dif)**2 )
        self.angle = np.arctan(y_dif/x_dif)
        # return cx,cy,w,h,angle            
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
