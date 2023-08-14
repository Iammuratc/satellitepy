from satellitepy.utils.path_utils import get_project_folder
from pathlib import Path

import json
import numpy as np
import cv2 as cv
import math
import utm

def main():
    imgs = []
    img_files = []

    folder_path = Path("/mnt/2tb-0/satellitepy/data/Gaofen")
    sequences_file = folder_path / "sequences.json"

    no_duplicates_file_train = folder_path / "train/recognition/no_duplicates_test.json"
    no_duplicates_file_val = folder_path / "val/recognition/no_duplicates_test.json"
    no_duplicates_file_test = folder_path / "test/recognition/no_duplicates_test.json"
    

    f = open(str(sequences_file), "r")
    sequences = json.loads(f.read())
    f.close()

    f1 = open(str(no_duplicates_file_train), "r")
    no_duplicates_train = json.loads(f1.read())
    f1.close()

    f2 = open(str(no_duplicates_file_val), "r")
    no_duplicates_val = json.loads(f2.read())
    f2.close()

    f3 = open(str(no_duplicates_file_test), "r")
    no_duplicates_test = json.loads(f3.read())
    f3.close()

    my_list = []

    merged_dict = {
        "id" : 0,
        "image_path" : "",
        "image_width" : 0,
        "image_height" : 0,
        "spatial_resolution_dy" : -0.81,
        "spatial_resolution_dx" :  0.81,
        "ground_truth" : []
    }

    label_dict = {
        "pixel_position" : "",
        "class" : ""
    }

    for i in range(0, len(sequences)):
        current_dict = {
            "id" : 0,
            "image_path" : "",
            "image_width" : 0,
            "image_height" : 0,
            "spatial_resolution_dy" : -0.81,
            "spatial_resolution_dx" :  0.81,
            "ground_truth" : []
        }

        current_label = {
            "pixel_position" : "",
            "class" : ""
        }


        ids = [int(str(folder_path / x['relative_path']).split("/")[-1].split(".")[0])-1 for x in sequences[i]['images']]
        img_files = [str(folder_path / x['relative_path']) for x in sequences[i]['images']]

        if sequences[i]['images'][0]['relative_path'].startswith('train'):
            no_duplicates = no_duplicates_train
        elif sequences[i]['images'][0]['relative_path'].startswith('val'):
            no_duplicates = no_duplicates_val
        else:
            no_duplicates = no_duplicates_test

        min_x = 100000000
        max_x = 0
        min_y = 100000000
        max_y = 0

        xy_corner = []

        for id in ids:

            poly = no_duplicates[id]['base_images'][0]['wkt_epsg_4326']
            poly = poly.replace("POLYGON ((", "").replace("))", "").split(", ")

            wgs_bbox = [[float(y) for y in x.split(" ")] for x in poly]

            s_res_y = 1 / -no_duplicates[id]['base_images'][0]['spatial_resolution_dy']
            s_res_x = 1 / no_duplicates[id]['base_images'][0]['spatial_resolution_dx']

            # xy_bbox = [[int(x[0] * s_res_x), int(x[1] * s_res_y)] for x in wgs_bbox]
            xy_bbox = [[int(y*1.23) for y in utm.from_latlon(x[1], x[0])[0:2]] for x in wgs_bbox]

            xy_min = [min([x[0] for x in xy_bbox]), min([x[1] for x in xy_bbox])]

            min_x = min(min_x, abs(xy_min[0]))
            max_x = max(max_x, abs(xy_min[0]))
            min_y = min(min_y, abs(xy_min[1]))
            max_y = max(max_y, abs(xy_min[1]))

            xy_corner.append(xy_min)

        max_x = max_x - min_x + 1024
        max_y = max_y - min_y + 1024

        stitch = np.zeros((max_y, max_x, 3)).astype('uint8')

        for j in range(len(img_files)):

            # img = cv.imread(img_files[j])
            x = xy_corner[j][0] - min_x
            y = xy_corner[j][1] - np.sign(xy_corner[j][1]) * min_y

            # if xy_corner[j][1] < 0:
            #     stitch[-y:-y+1024, x:x+1024, 0:3] = img
            # else:
            #     stitch[max_y - y - 1024:max_y - y, x:x+1024, 0:3] = img

            ground_truth = no_duplicates[j]['base_images'][0]['ground_truth']

            for label in ground_truth:
                bbox = [[int(y) for y in x.split(" ")] for x in label['pixel_position'].replace("POLYGON ((", "").replace("))", "").split(", ")]

                if xy_corner[j][1] < 0:
                    bbox = [[int(x + corner[0]), int(-y + corner[1])] for corner in bbox]
                else: 
                    bbox = [[int(x + corner[0]), int((max_y - y - 1024) + corner[1])] for corner in bbox]

                current_label['pixel_position'] = bbox
                current_label['class'] = label['class']

                current_dict['ground_truth'].append(current_label)


        out_file = get_project_folder() / f"in_folder/stitch_test_coords/img_{i}.tif"
        # cv.imwrite(str(out_file), stitch)
        print(f"Image {i} successfully merged into ", out_file)

        # setting up the dict
        current_dict['id'] = i
        current_dict['image_path'] = str(out_file)
        current_dict['image_width'] = max_x
        current_dict['image_height'] = max_y
        current_dict['spatial_resolution_dy'] = -0.81
        current_dict['spatial_resolution_dx'] =  0.81

        print(len(json.dumps(current_dict)))

        my_list.append(current_dict)

        f1 = open(str(get_project_folder() / "in_folder/my_json_file.json"), "w")
        f1.write(json.dumps(my_list, indent=4))
        f1.close()
        
        

if __name__ == '__main__':
    main()

    # following function provides the sorted json files for no dublicates
    # no_duplicates_val = sorted(no_duplicates_val, key = lambda item: int(Path(item['base_images'][0]['image_path']).stem.split(".")[0]))

