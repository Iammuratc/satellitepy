import numpy as np
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
        if is_corners:
            corners = kwargs['corners']
            if isinstance(corners, list):
                self.corners = np.array(corners)
            elif isinstance(corners, np.ndarray):
                self.corners = corners
            else:
                raise Exception('Corners have to be either list or numpy array')
            # self.params = self.get_params_cv2()
            self.params = self.get_params()
        elif is_params:
            # params = np.array(kwargs['params'])
            params = kwargs['params']
            if isinstance(params, list):
                params = np.array(params)
            elif isinstance(params, np.ndarray):
                self.params = params            
            else:
                raise Exception('Params have to be either list or numpy array')
            self.params = params
            self.corners=self.get_corners()
        else:
            raise Exception('Either corners or params have to be defined to create a BBox instance')
            return 0

    def get_neighbor_corner_dif(self,corners,i):
        i_0 = i
        i_1 = i + 1

        # print(f'y points {corners[i_0][1]},{corners[i_1][1]}')
        x_dif = corners[i_1][0] - corners[i_0][0]
        y_dif = corners[i_1][1] - corners[i_0][1]
        y_sum = corners[i_0][1] + corners[i_1][1] 
        return x_dif, y_dif, y_sum


    def get_corners(self):
        cx,cy,h,w,angle = self.params

        corners = np.array([
            [cx+w/2.0,cy+h/2.0],
            [cx+w/2.0,cy-h/2.0],
            [cx-w/2.0,cy-h/2.0],
            [cx-w/2.0,cy+h/2.0]])


        corners = self.rotate_corners(corners=corners,angle=angle)
        # print(corners)
        return corners

    def get_params_cv2(self):
        (x_c, y_c), (width, height), angle = cv2.minAreaRect(self.corners)
        return x_c, y_c, width, height, angle

    def get_params(self):
        cx = np.sum(self.corners[:, 0]) / 4.0
        cy = np.sum(self.corners[:, 1]) / 4.0

        # def x_dif(i_0, i_1): return self.corners[i_0, 0] - self.corners[i_1, 0]
        # def y_dif(i_0, i_1): return self.corners[i_0, 1] - self.corners[i_1, 1]
        direction = self.get_direction()

        x_dif_0_1, y_dif_0_1,_ = self.get_neighbor_corner_dif(self.corners,i=0)
        x_dif_1_2, y_dif_1_2,_ = self.get_neighbor_corner_dif(self.corners,i=1)
        if direction == 'clockwise':
            h = np.sqrt(x_dif_0_1**2 + y_dif_0_1**2)
            w = np.sqrt(x_dif_1_2**2 + y_dif_1_2**2)
            arctan2_angle = np.arctan2(y_dif_0_1,x_dif_0_1)
        elif direction == 'counter-clockwise':
            w = np.sqrt(x_dif_0_1**2 + y_dif_0_1**2)
            h = np.sqrt(x_dif_1_2**2 + y_dif_1_2**2)
            arctan2_angle = np.arctan2(y_dif_1_2,x_dif_1_2)

        # angle = np.arctan(y_dif(0, 1) / x_dif(0, 1))
        # print(f'arctan result : {angle}')
        # print(f'arctan2 result: {arctan2_angle}')
        # direction = self.get_direction(self.corners)
        return [cx, cy, h, w, arctan2_angle] # 2*np.pi-

    # def get_corners(self):
    def switch_direction(self,bbox):
        bbox = np.array(bbox)    
        bbox_copy = bbox.copy()
        coord_1 = bbox_copy[1, :]
        coord_3 = bbox_copy[3, :]
        bbox[3, :] = coord_1
        bbox[1, :] = coord_3
        return bbox
    def get_direction(self,corners=None):
        '''
        Get the direction of corners (clockwise or counter-clockwise)
        '''
        if corners==None:
            corners = self.corners

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
            orth_angle = -params_angle
        elif direction=='counter-clockwise':
            orth_angle =  3 * np.pi / 2 - params_angle
        return orth_angle

    # def get_orthogonal_bbox(self,angle):
    def rotate_corners(self,angle,corners=None):
        if corners is None:
            corners = self.corners
        # angle = -self.get_orth_angle()
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        # o = np.atleast_2d((self.cx, self.cy))
        o = np.atleast_2d((self.params[0], self.params[1]))
        # bbox = np.atleast_2d(self.bbox)
        # return np.squeeze((R @ (self.corners.T - o.T) + o.T).T)
        return np.squeeze((R @ (corners.T - o.T) + o.T).T)

    # def get_orthogonal_bbox_by_limits(self, corners, return_params):
    #     x_coords = corners[:, 0]
    #     y_coords = corners[:, 1]
    #     x_coord_min, x_coord_max = np.amin(x_coords), np.amax(x_coords)
    #     y_coord_min, y_coord_max = np.amin(y_coords), np.amax(y_coords)

    #     if return_params:
    #         h = y_coord_max - y_coord_min
    #         w = x_coord_max - x_coord_min
    #         return [x_coord_min + w / 2.0, y_coord_min + h / 2.0, w, h]
    #     else:
    #         return [[x_coord_min, y_coord_max],
    #                 [x_coord_max, y_coord_max],
    #                 [x_coord_max, y_coord_min],
    #                 [x_coord_min, y_coord_min]]

    @staticmethod
    def plot_bbox(corners, ax, c='b', s=15, instance_name=None):
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        for i, coord in enumerate(corners):
            # PLOT BBOX
            ax.plot([corners[i - 1][0], coord[0]],
                    [corners[i - 1][1], coord[1]], c=c)
        if instance_name is not None:
            x_min, x_max, y_min, y_max = BBox.get_bbox_limits(corners)
            ax.text(x=(x_max+x_min)/2,y=(y_max+y_min)/2,s=instance_name, fontsize=8, color='r', backgroundcolor='black', alpha=1) #  fontweight='bold',
        # ax.scatter(corners[0][0], corners[0][1], c='r', s=s)
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