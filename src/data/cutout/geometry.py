import shapely.geometry
import shapely.affinity
import numpy as np
from descartes import PolygonPatch
import cv2


# TODO: Detection and recognition functionilities are confusing. Remove
# class attributes, and call methods (not attributes) from
# RecognitionPatch and DetectionPatch
class Bbox:
    '''
    This class does the followings:
    - Parametrization of bbox corner points, i.e., corner points >> center_x, center_y, height, length, rotation angle
    - Params to points, i.e.,  center_x, center_y, height, length, rotation angle >> corner points
    - Rotation calculations of bounding boxes

    '''

    def __init__(self, bbox):
        if isinstance(bbox, list):
            bbox = np.array(bbox)

        if bbox.shape == (4, 2):
            # self.bbox = bbox
            self.bbox_corners = bbox
            # self.cx, self.cy, self.h, self.w, self.angle = self.get_bbox_params()
            self.bbox_params = self.get_bbox_params()
            # self.orthogonal_bbox = self.get_orthogonal_bbox()

        elif bbox.shape == (1, 5):
            self.bbox_params = bbox

        else:
            raise Exception('bbox shape should be (4,2) for the corner definition OR (1,5) for the param definition')
            return 0

    def get_params(self):
        cx = np.sum(self.bbox_corners[:, 0]) / 4.0
        cy = np.sum(self.bbox_corners[:, 1]) / 4.0

        # def x_dif(i_0, i_1): return self.bbox_corners[i_0, 0] - self.bbox_corners[i_1, 0]
        # def y_dif(i_0, i_1): return self.bbox_corners[i_0, 1] - self.bbox_corners[i_1, 1]

        x_dif = lambda (i_0, i_1): self.bbox_corners[i_0, 0] - self.bbox_corners[i_1, 0]
        y_dif = lambda (i_0, i_1): self.bbox_corners[i_0, 1] - self.bbox_corners[i_1, 0]

        h = np.sqrt(x_dif(0, 1)**2 + y_dif(0, 1)**2)
        w = np.sqrt(x_dif(1, 2)**2 + y_dif(1, 2)**2)

        angle = np.arctan(y_dif(0, 1) / x_dif(0, 1))
        return [cx, cy, h, w, angle]

    def get_corners(self):
        

    def get_atan2(self):  # ,img_shape):
        '''
        Get the rotation matrix to rotate the airplane upwards
        '''
        # y_max, x_max, _ = img_shape
        # print(np.rad2deg(self.angle))
        if (self.bbox_corners[0, 1] - self.bbox_corners[1, 1] >=0) and (self.bbox_corners[0, 0] - self.bbox_corners[1, 0] >= 0):
            # print('q1')
            angle = 3 * np.pi / 2 + self.angle
        elif (self.bbox_corners[0, 1] - self.bbox_corners[1, 1] >= 0) and (self.bbox_corners[0, 0] - self.bbox_corners[1, 0] <= 0):
            # print('q2')
            angle = np.pi / 2 + self.angle
        elif (self.bbox_corners[0, 1] - self.bbox_corners[1, 1] <= 0) and (self.bbox_corners[0, 0] - self.bbox_corners[1, 0] <= 0):
            # print('q3')
            angle = np.pi / 2 + self.angle
        elif (self.bbox_corners[0, 1] - self.bbox_corners[1, 1] <= 0) and (self.bbox_corners[0, 0] - self.bbox_corners[1, 0] >= 0):
            # print('q4')
            angle = 3 * np.pi / 2 + self.angle
        # M = cv2.getRotationMatrix2D((y_max/2, x_max/2), np.rad2deg(angle), 1.0)

        # return M,angle
        return angle

    def get_orthogonal_bbox(self):

        angle = -self.get_atan2()
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        # o = np.atleast_2d((self.cx, self.cy))
        o = np.atleast_2d((self.bbox_params[0], self.bbox_params[1]))
        # bbox = np.atleast_2d(self.bbox)
        return np.squeeze((R @ (self.bbox_corners.T - o.T) + o.T).T)

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
    def plot_bbox(bbox_corners, ax, c, s=15):
        for i, coord in enumerate(bbox):
            # PLOT BBOX
            ax.plot([bbox[i - 1][0], coord[0]],
                    [bbox[i - 1][1], coord[1]], c=c)
        ax.scatter(bbox[0][0], bbox[0][1], c='r', s=s)
        return ax

    @staticmethod
    def get_bbox_limits(bbox):
        # print(bbox)
        x_min = np.amin(bbox[:, 0])
        x_max = np.amax(bbox[:, 0])
        y_min = np.amin(bbox[:, 1])
        y_max = np.amax(bbox[:, 1])
        return np.array([x_min, x_max, y_min, y_max]).astype(int)


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
    P = shapely.wkt.loads(
        'POLYGON ((51.0 3.0, 51.3 3.61, 51.3 3.0, 51.6 4.0, 51.0 3.0))')
    # print(np.array(P.exterior.coords))
    bbox = np.array(P.exterior.coords)[0:4, :]
    print(bbox)
