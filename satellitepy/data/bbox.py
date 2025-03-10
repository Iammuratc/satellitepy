import numpy as np
import cv2


class BBox:
    """
    This class does the followings:
    - Parametrization of bbox corner points, i.e., corner points >> center_x, center_y, width, height, rotation angle
    - Params to points, i.e.,  center_x, center_y, width, height, rotation angle >> corner points
    - Rotation calculations of bounding boxes
    """

    def __init__(self, **kwargs):

        if 'corners' in kwargs.keys():
            corners = kwargs['corners']

            if isinstance(corners, list):
                self.corners = np.array(corners)
            elif isinstance(corners, np.ndarray):
                self.corners = corners
            else:
                raise Exception('Corners have to be either list or numpy array')
            # print(self.corners)
            if np.shape(self.corners)[-2:] != (4, 2):
                raise Exception('The bbox in corner parametrization must end in shape (4,2) but has shape:',
                                np.shape(corners))
            self.params = self.get_params()

        elif 'diamond_corners' in kwargs.keys():
            self.diamond_corners = kwargs['diamond_corners']
            if isinstance(self.diamond_corners, list):
                self.diamond_corners = np.array(self.diamond_corners)
            elif isinstance(self.diamond_corners, np.ndarray):
                pass
            else:
                raise Exception('Diamond corners must be either list or numpy array.')
            if np.shape(self.diamond_corners)[-2:] != (4, 2):
                raise Exception('The diamond bbox in corner parametrization must end in shape (4,2) but has shape:',
                                np.shape(self.diamond_corners))
            self.corners = self.convert_diamond_corners_to_corners(self.diamond_corners)
            self.params = self.get_params()

        elif 'params' in kwargs.keys():
            params = kwargs['params']

            if isinstance(params, list):
                params = np.array(params)
            elif isinstance(params, np.ndarray):
                self.params = params
            else:
                raise Exception('Params have to be either list or numpy array')

            if np.shape(params)[-1] != 5:
                raise Exception('The bbox in parameter parametrization must end in shape (5) but has shape:',
                                np.shape(params))
            self.params = params
            self.corners = self.get_corners()

        else:
            raise Exception('Either corners or params have to be defined to create a BBox instance')

    def convert_diamond_corners_to_corners(self, diamond_corners):
        """
        convert the diamond corners into the normal corner representation.
        the corners are in the clockwise orientation with the angle pointing through the face between corners 0 and 1.
        """
        A, B, C, D = diamond_corners
        vecBD = tuple(np.subtract(D, B))
        middle = tuple(np.add(B, np.divide(vecBD, 2)))
        vecToC = tuple(np.subtract(C, middle))
        vecToA = tuple(np.subtract(A, middle))

        corners = np.array([np.add(D, vecToA).tolist(), np.add(D, vecToC).tolist(), np.add(B, vecToC).tolist(),
                            np.add(B, vecToA).tolist()])
        return corners

    def get_neighbor_corner_dif(self, corners, i):
        i_0 = i
        i_1 = i + 1

        x_dif = corners[i_1][0] - corners[i_0][0]
        y_dif = corners[i_1][1] - corners[i_0][1]
        y_sum = corners[i_0][1] + corners[i_1][1]
        return x_dif, y_dif, y_sum


    def get_corners(self):
        """
        convert the angle representation into the corner representation.
        the corners are in the clockwise orientation with the angle pointing through the face between corners 0 and 1.
        """

        # Extract the centerposition, width, height and angle from the Boxes array
        x, y, h, w, angle = self.params

        # Calculate the four corners of the Boxes(first one is to left for angle=0, rest follow clockwise)
        x1, y1 = x - np.cos(angle) * h / 2 + np.sin(angle) * w / 2, y + np.sin(angle) * h / 2 + np.cos(angle) * w / 2

        x2, y2 = x + np.cos(angle) * h / 2 + np.sin(angle) * w / 2, y - np.sin(angle) * h / 2 + np.cos(angle) * w / 2

        x3, y3 = x + np.cos(angle) * h / 2 - np.sin(angle) * w / 2, y - np.sin(angle) * h / 2 - np.cos(angle) * w / 2

        x4, y4 = x - np.cos(angle) * h / 2 - np.sin(angle) * w / 2, y + np.sin(angle) * h / 2 - np.cos(angle) * w / 2

        # Fill in the Corners Array with the previously calculated corners

        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    def get_params_cv2(self):
        (x_c, y_c), (width, height), angle = cv2.minAreaRect(self.corners)
        return x_c, y_c, width, height, angle

    def get_params(self):
        """
        This function converts the bbox from the corner representation into the parameter representation.
        'cx' and 'cy' are the center of the bbox,
        'h' and 'w' are the height and width of the bbox and
        'angle' is the angle of the bbox
        """

        c1, c2, c3, c4 = self.corners[0, :], self.corners[1, :], self.corners[2, :], self.corners[3, :]

        cx = (c1[0] + c2[0] + c3[0] + c4[0]) / 4
        cy = (c1[1] + c2[1] + c3[1] + c4[1]) / 4

        width = (np.linalg.norm(c1 - c2, 2, -1) + np.linalg.norm(c3 - c4, 2, -1)) / 2
        height = (np.linalg.norm(c1 - c4, 2, -1) + np.linalg.norm(c2 - c3, 2, -1)) / 2

        direction = self.get_direction()

        if direction == 'clockwise':
            angle = np.arctan2(c1[1] - c2[1], c2[0] - c1[0])
            if angle != 0:
                angle = angle + np.pi - np.sign(angle) * np.pi

        elif direction == 'counter-clockwise':
            angle = np.arctan2(c1[1] - c2[1], c1[0] - c2[0])
            if angle != 0:
                angle = angle + np.pi - np.sign(angle) * np.pi

        return [cx, cy, width, height, angle]

    def switch_direction(self, bbox):
        bbox = np.array(bbox)
        bbox_copy = bbox.copy()
        coord_1 = bbox_copy[1, :]
        coord_3 = bbox_copy[3, :]
        bbox[3, :] = coord_1
        bbox[1, :] = coord_3
        return bbox

    def get_direction(self, corners=None):
        """
        compute the direction of the corners by utilizing the cross-product between the two vectors spanning three corners, by pretending they are two vectors in three dimensions (both in the x,y plane)
        negative value of z component -> clockwise, positive value of z-component -> couter clockwise, z=0 -> the first two points are identical -> there is no propper bounding box
        """
        if corners is None:
            corners = self.corners

        corners_expanded = np.concatenate([corners, np.zeros([4, 1])], -1)
        cross = np.cross(corners_expanded[1] - corners_expanded[0], corners_expanded[1] - corners_expanded[2])

        if cross[2] <= 0:
            return 'counter-clockwise'
        else:
            return 'clockwise'

    def get_orth_angle(self, direction=None):
        """
        Get the orthogonal angle to rotate the airplane upwards
        """

        params_angle = self.params[-1]

        if direction == None:
            direction = self.get_direction(self.corners)

        if direction == 'clockwise':
            orth_angle = -params_angle
        elif direction == 'counter-clockwise':
            orth_angle = 3 * np.pi / 2 - params_angle
        return orth_angle

    def rotate_corners(self, angle, corners=None):
        if corners is None:
            corners = self.corners

        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

        o = np.atleast_2d((self.params[0], self.params[1]))
        return np.squeeze((rot @ (corners.T - o.T) + o.T).T)

    @staticmethod
    def plot_bbox(corners, ax, c='b', instance_name=None):
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        for i, coord in enumerate(corners):
            ax.plot([corners[i - 1][0], coord[0]],
                    [corners[i - 1][1], coord[1]], c=c)
        if instance_name is not None:
            x_min, x_max, y_min, y_max = BBox.get_bbox_limits(corners)
            ax.text(x=(x_max + x_min) / 2, y=(y_max + y_min) / 2 - 5, s=instance_name, fontsize=8, color='r', alpha=1,
                    horizontalalignment='center', verticalalignment='bottom')
        return ax

    @staticmethod
    def draw_bbox_to_img(img, corners, color=(255, 0, 0), thickness=1):
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        img = cv2.polylines(img, corners.astype(np.int32), isClosed=True, color=color, thickness=thickness)
        return img

    @staticmethod
    def get_bbox_limits(corners):
        if not isinstance(corners, np.ndarray):
            corners = np.array(corners)
        x_min = np.amin(corners[:, 0])
        x_max = np.amax(corners[:, 0])
        y_min = np.amin(corners[:, 1])
        y_max = np.amax(corners[:, 1])
        return np.array([x_min, x_max, y_min, y_max]).astype(int)

    @staticmethod
    def get_hbb_from_obb(obb):
        x_coords = [obb[0][0], obb[1][0], obb[2][0], obb[3][0]]
        y_coords = [obb[0][1], obb[1][1], obb[2][1], obb[3][1]]

        max_x = max(x_coords)
        min_x = min(x_coords)

        max_y = max(y_coords)
        min_y = min(y_coords)

        hbb = [[min_x, max_y], [min_x, min_y], [max_x, min_y], [max_x, max_y]]
        return hbb


def le90_to_corners(params):
    """
    Convert bounding box parameters (xc, yc, height, width, angle) in 'le90' format 
    to the four corners of the bounding box, where height is along the y-axis and
    width is along the x-axis.

    Args:
        xc (float): x-coordinate of the bounding box center.
        yc (float): y-coordinate of the bounding box center.
        height (float): Height of the bounding box.
        width (float): Width of the bounding box.
        angle (float): Rotation angle in degrees, in the range [-90, 90].

    Returns:
        np.ndarray: An array of shape (4, 2), representing the four corner points 
                    in clockwise order starting from the top-left.
    """
    # Ensure height is the longer dimension
    xc, yc, width, height, angle = params

    # if height < width:
    #     height, width = width, height  # Swap height and width
    #     angle += 90                   # Adjust the angle
    #     if angle > 90:
    #         angle -= 180              # Keep angle in [-90, 90]
    
    # Convert angle to radians
    theta = np.radians(angle)
    # theta = angle
    # Compute cosine and sine of the rotation angle
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Half dimensions
    half_height = height / 2.0
    half_width = width / 2.0

    # Compute the unrotated corner offsets relative to the center
    offsets = np.array([
        [-half_width, -half_height],  # Top-left corner
        [ half_width, -half_height],  # Top-right corner
        [ half_width,  half_height],  # Bottom-right corner
        [-half_width,  half_height],  # Bottom-left corner
    ])

    # Apply rotation using the rotation matrix
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta],
    ])

    # Rotate each corner
    rotated_offsets = offsets @ rotation_matrix.T

    # Translate the rotated offsets to the center position
    corners = rotated_offsets + np.array([xc, yc])

    return corners