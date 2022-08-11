import cv2
import numpy as np
import json

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
    def __init__(self,settings,dataset_part):
        self.project_folder = settings['project_folder']
        self.dataset_id=settings['dataset']['id']
        self.dataset_part=dataset_part # train, test or val
        self.dataset_name=settings['dataset']['name']

        ### EX: /home/murat/Projects/airplane_recognition/data/Gaofen/train/recognition/no_duplicates.json        
        self.json_file_path = settings['patch'][dataset_part]['json_file_path']

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