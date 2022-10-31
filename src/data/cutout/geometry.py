import shapely.geometry
import shapely.affinity
import numpy as np
from descartes import PolygonPatch
import cv2


# TODO: 
#   

class BBox:
    '''
    This class does the followings:
    - Parametrization of bbox corner points, i.e., corner points >> center_x, center_y, height, length, rotation angle
    - Params to points, i.e.,  center_x, center_y, height, length, rotation angle >> corner points
    - Rotation calculations of bounding boxes

    '''

    def __init__(self, **kwargs):
        is_corners = 'corners' in kwargs.keys() 
        is_params = 'params' in kwargs.keys() 


        if (not is_corners) and (not is_params):
            raise Exception('either corners or params have to be defined to create a BBox instance')
            return 0

        # if bbox.shape == (4, 2):
        if is_corners:
            corners = kwargs['corners']
            if isinstance(corners, list):
                self.corners = np.array(corners)
            elif isinstance(corners, np.ndarray):
                self.corners = corners            
            else:
                raise Exception('Corners have to be either list or numpy array')
            self.params = self.get_params()

        # elif bbox.shape == (1, 5):
        elif is_params:
            params = np.array(kwargs['params'])
            if isinstance(params, list):
                params = np.array(params)
            elif isinstance(params, np.ndarray):
                self.params = params            
            else:
                raise Exception('Params have to be either list or numpy array')
            self.params = params

    def get_neighbor_corner_dif(self,corners,i):
        i_0 = i
        i_1 = i + 1

        # print(f'y points {corners[i_0][1]},{corners[i_1][1]}')
        x_dif = corners[i_1][0] - corners[i_0][0]
        y_dif = corners[i_1][1] - corners[i_0][1]
        y_sum = corners[i_0][1] + corners[i_1][1] 
        return x_dif, y_dif, y_sum



    def get_params(self):
        cx = np.sum(self.corners[:, 0]) / 4.0
        cy = np.sum(self.corners[:, 1]) / 4.0

        # def x_dif(i_0, i_1): return self.corners[i_0, 0] - self.corners[i_1, 0]
        # def y_dif(i_0, i_1): return self.corners[i_0, 1] - self.corners[i_1, 1]
        x_dif_h, y_dif_h,_ = self.get_neighbor_corner_dif(self.corners,i=0)
        x_dif_w, y_dif_w,_ = self.get_neighbor_corner_dif(self.corners,i=1)
        h = np.sqrt(x_dif_h**2 + y_dif_h**2)
        w = np.sqrt(x_dif_w**2 + y_dif_w**2)

        # angle = np.arctan(y_dif(0, 1) / x_dif(0, 1))
        # print(f'arctan result : {angle}')
        arctan2_angle = np.arctan2(y_dif_h,x_dif_h)
        # print(f'arctan2 result: {arctan2_angle}')
        # direction = self.get_direction(self.corners)
        return [cx, cy, h, w, arctan2_angle]

    # def get_corners(self):
    
    def get_direction(self,corners):
        '''
        Get the direction of corners (clockwise or counter-clockwise)
        '''

        sum_over_edges = 0
        corners = np.append(corners,[corners[0]],axis=0) # append the first element to enable the sum over ege calculation
        for i in range(len(corners)-1):
            x_dif, _, y_sum = self.get_neighbor_corner_dif(corners,i)
            # print(f'x dif and y sum: {x_dif},{y_sum}')
            sum_over_edges += x_dif*y_sum
        # print(f'sum over edges is {sum_over_edges}')
        if sum_over_edges >= 0:
            return 'counter-clockwise'
        else:
            return 'clockwise'


    def get_orth_angle(self,direction=None):  # ,img_shape):
        '''
        Get the orthogonal angle to rotate the airplane upwards
        '''

        params_angle=self.params[-1]

        if direction == None:
            direction=self.get_direction(self.corners)

        # print(f'direction is {direction}')
        if direction=='clockwise':
            # orth_angle = np.pi / 2 + params_angle
            orth_angle = params_angle
        elif direction=='counter-clockwise':
            orth_angle =  3 * np.pi / 2 + params_angle
        return orth_angle

    def get_orthogonal_bbox(self):

        angle = -self.get_orth_angle()
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        # o = np.atleast_2d((self.cx, self.cy))
        o = np.atleast_2d((self.params[0], self.params[1]))
        # bbox = np.atleast_2d(self.bbox)
        return np.squeeze((R @ (self.corners.T - o.T) + o.T).T)

    def get_orthogonal_bbox_by_limits(self, bbox, return_params):
        x_coords = bbox[:, 0]
        y_coords = bbox[:, 1]
        x_coord_min, x_coord_max = np.amin(x_coords), np.amax(x_coords)
        y_coord_min, y_coord_max = np.amin(y_coords), np.amax(y_coords)

        if return_params:
            h = y_coord_max - y_coord_min
            w = x_coord_max - x_coord_min
            return [x_coord_min + w / 2.0, y_coord_min + h / 2.0, w, h]
        else:
            return [[x_coord_min, y_coord_max],
                    [x_coord_max, y_coord_max],
                    [x_coord_max, y_coord_min],
                    [x_coord_min, y_coord_min]]

    @staticmethod
    def plot_bbox(corners, ax, c, s=15):
        for i, coord in enumerate(corners):
            # PLOT BBOX
            ax.plot([corners[i - 1][0], coord[0]],
                    [corners[i - 1][1], coord[1]], c=c)
        ax.scatter(corners[0][0], corners[0][1], c='r', s=s)
        return ax

    @staticmethod
    def get_bbox_limits(corners):
        # print(bbox)
        x_min = np.amin(corners[:, 0])
        x_max = np.amax(corners[:, 0])
        y_min = np.amin(corners[:, 1])
        y_max = np.amax(corners[:, 1])
        return np.array([x_min, x_max, y_min, y_max]).astype(int)


if __name__ == "__main__":
    import numpy as np
    import shapely.wkt

    import matplotlib.pyplot as plt
    import json
    import os
    import random
    import cv2

    corners = np.array([[43.0, 46.0], [47.0, 87.0], [86.0, 83.0], [81.0, 41.0]])

    my_bbox = BBox(corners=corners)
    orth_bbox = my_bbox.get_orthogonal_bbox()
    # print(rect.bbox)

    # print(orth_bbox)

    fig,ax = plt.subplots(1)
    ax.set_ylim([0,256])
    ax.set_xlim([0,256])

    my_bbox.plot_bbox(ax=ax,corners=my_bbox.corners,c='y')

    my_bbox.plot_bbox(ax=ax,corners=orth_bbox,c='b')

    plt.show()

    # P = np.asarray(shapely.wkt.loads('POLYGON ((51.0 3.0, 51.3 3.61, 51.3 3.0, 51.0 3.0))'))
    # P = shapely.wkt.loads(
    #     'POLYGON ((51.0 3.0, 51.3 3.61, 51.3 3.0, 51.6 4.0, 51.0 3.0))')
    # # print(np.array(P.exterior.coords))
    # bbox = np.array(P.exterior.coords)[0:4, :]
    # print(bbox)
