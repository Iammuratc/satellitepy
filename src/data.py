import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import os
import json
import math
import geometry
# from utilities import show_sample
import matplotlib.pyplot as plt
import requests

##NOTES: y axis of matplotlib figures are inverted, so the airplanes will be actually facing downwards, pay attention at the new datasets 

class DataDem:
    """
        This class is to create the json file using DEM
    """
    def __init__(self,dataset_id,dataset_part,dataset_name):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_id=dataset_id
        self.dataset_part=dataset_part # train, test or val
        self.dataset_name=dataset_name
        
        self.data_folder = f"{self.project_folder}/DATA"

        self.json_file_path = f'{self.data_folder}/{self.dataset_name}/{self.dataset_part}/no_duplicates.json'

        self.data = self.remove_duplicates()


    def remove_duplicates(self):
        """
            Remove the duplicated objects by using the object geographies
        """

        ### Store the json file to avoid accessing the database everytime
        
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as fp:
                result = json.load(fp)
            return result        
        else:
            print(f"Json file will be saved at: {self.json_file_path}")

        ### Base URL
        dqp_url = "http://localhost:4000/api/v1"

        ### IMAGE SEQUENCES
        params = {
            "dataset_id": self.dataset_id,
            "include_nested": True
            }
        sequences = requests.get(f"{dqp_url}/image-sequence", params=params).json()


        result = []

        for s in sequences:
            # Sequence of images at a specific location (e.g. at an airport)
            
            # Get list of object geometries to find the duplicates 
            obj_geos = []

            for i,base_image in enumerate(s["base_images"]):
                res = {
                "sequence_id": s["external_id"],
                "base_images": []
                }    

                # SWITCH DEM IMAGE PATH TO LOCAL IMAGE PATH
                image_path_dem = base_image["image_path"][10:]
                image_path = f"{self.data_folder}/{image_path_dem}"

                # READ ONLY TRAIN, TEST OR VAL IMAGES
                path = os.path.normpath(image_path)
                base_image_part = path.split(os.sep)[-3]
                if base_image_part != self.dataset_part:
                    continue

                # APPEND THE BASE IMAGES
                res['base_images'].append({ 
                                            "id": base_image['id'],
                                            "image_id": base_image["external_id"],
                                            "image_path": image_path,
                                            "image_width": base_image["image_width"],
                                            "image_height": base_image["image_height"],
                                            "spatial_resolution_dy": base_image["spatial_resolution_dy"],
                                            "spatial_resolution_dx": base_image["spatial_resolution_dx"],
                                            "orig_projection": base_image["srs"],
                                            "orig_wkt": base_image["geometry"],
                                            "wkt_epsg_4326": base_image["geography"],
                                            "ground_truth": []
                                            })

                params = {
                    "base_image.id": base_image['id'],
                    "include_nested": True
                }
                objects = requests.get(f"{dqp_url}/ground-truth-object", params=params).json()

                for obj in objects:
                    obj_geo = obj["object_geographies"][0]["geometry"]

                    # If the object is not stored yet
                    if obj_geo not in obj_geos:
                        obj_geos.append(obj_geo)
                        res['base_images'][0]['ground_truth'].append({
                                                                    "pixel_position": obj["object_geographies"][0]["image_geometry"],
                                                                    "orig_projection": obj["object_geographies"][0]["srs"],
                                                                    "orig_wkt": obj["object_geographies"][0]["geometry"],
                                                                    "wkt_epsg_4326": obj["object_geographies"][0]["geography"],
                                                                    "class": obj["object_types"][0]["name"]
                                                                    })

                result.append(res)


        with open(self.json_file_path, 'w') as fp:
            json.dump(result, fp, indent=4)

        return result


class Data:
    """
        This class is created to read the original json file (i.e. sequences.json) of the Gaofen dataset
    """
    def __init__(self,dataset_name):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = f"{self.project_folder}/DATA/Gaofen"
        self.dataset_name = dataset_name
        self.json_file = json.load(open(f"{self.data_folder}/sequences.json",'r'))


    def get_img_paths(self):
        img_paths = {}
        for i in self.json_file:
            image_dicts = i['images']
            for image_dict in image_dicts:
                relative_path = image_dict['relative_path']
                relative_path_split = relative_path.split('/')
                dataset_part = relative_path_split[0]
                img_name = relative_path_split[-1].split('.')[0]
                if dataset_part == self.dataset_name:
                    img_paths[img_name] = f"{self.data_folder}/{relative_path}"
                    
        return img_paths 

    def get_label(self,label_path):
        label = {'bbox':[],'names':[]}
        root = ET.parse(label_path).getroot()

        ### IMAGE NAME
        file_name = root.findall('./source/filename')[0].text
        # img_name = file_name.split('.')[0]
         
        ### INSTANCE NAMES
        instance_names = root.findall('./objects/object/possibleresult/name')#[0].text
        for instance_name in instance_names:
            label['names'].append(instance_name.text)
        
        ### BBOX CCORDINATES
        point_spaces = root.findall('./objects/object/points')        
        for point_space in point_spaces:
            my_points = point_space.findall('point')[:4] # remove the last coordinate
            coords = []
            for my_point in my_points:
                #### [[[x1,y1],[x2,y2]],[[x1,y1]]]
                coord = []
                for point in my_point.text.split(','):
                    coord.append(float(point))
                coords.append(coord)
            label['bbox'].append(coords)
        return label#, img_name


    def plot_bboxes(self,img_path,label_path,save_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label = self.get_label(label_path)
        fig, ax = plt.subplots(1)

        ax.imshow(img)
        for bbox in label['bbox']:
            # print(bbox)
            bbox=np.array(bbox)
            rect = geometry.Rectangle(bbox)
            rect.plot_bbox(ax=ax,bbox=bbox,c='b')
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":

    dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
    dataset_name = 'Gaofen'
    dataset_part = 'val'


    data_dem = DataDem(dataset_id,dataset_part,dataset_name)

    ## Base URL
    # dqp_url = "http://localhost:4000/api/v1"
    # image_set_id = "b5a26782-b9d1-4f9b-bb81-e6ec35ff0592"
    # params = {
    #     # "id": image_set_id,
    #     "include_nested": False
    #     }
    # image_sets = requests.get(f"{dqp_url}/image-set",params=params).json()
    # print(image_sets)
    # sequences = requests.get(f"{dqp_url}/image-sequence", params=params).json()
    # print(sequences)


    ### GROUND TRUTH
    # my_dict = {
    #     "id" : "e097e5d7-3552-4018-8a41-660e7d85fba9",
    #     "image_id": "d1a5c470-d032-4309-afc4-903bbbcc198c",
    #         }
    # params = {
    #     "base_image.id": my_dict['id'],
    #     "include_nested": True
    #     }
    # objects = requests.get(f"{dqp_url}/ground-truth-object", params=params).json()
    # print(objects)
    ### IMAGE SEQUENCES
    # params = {
    #     "dataset_id": dataset_id,
    #     "include_nested": False
    #     }
    # sequences = requests.get(f"{dqp_url}/image-sequence", params=params).json()

    # for s in sequences:
    #     print(s["external_id"])
    #     for base_image in s['base_images']:
    #         print(base_image['image_path'])
