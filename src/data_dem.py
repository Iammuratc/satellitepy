import requests
import matplotlib.pyplot as plt
import cv2
import json


def remove_duplicates(dataset_id,read_json_file=False):
    """
        Remove the duplicated objects by using the object geographies
    """

    ### Store the json file to avoid accessing the database
    json_file_path = '/home/murat/Projects/airplane_recognition/DATA/Gaofen/train/no_duplicates.json'
    if read_json_file is False:
        print(f"Json file will be saved at: {json_file_path}")
    
    else:
        with open(json_file_path, 'r') as fp:
            result = json.load(fp)
        return result

    ### Base URL
    dqp_url = "http://localhost:4000/api/v1"

    ### IMAGE SEQUENCES
    params = {
        "dataset_id": dataset_id,
        "include_nested": True
        }
    sequences = requests.get(f"{dqp_url}/image-sequence", params=params).json()


    result = []

    for s in sequences:
        # Sequence of images at a specific location (e.g. at an airport)
        
        # Get list of object geometries to find the duplicates 
        obj_geos = []
    
        for base_image in s["base_images"]:

            res = {
                "sequence_id": s["external_id"],
                "base_images": []
            }

            res['base_images'].append({ 
                                        "id": base_image['id'],
                                        "image_id": base_image["external_id"],
                                        "image_path": base_image["image_path"],
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
                    # print(res['base_images'][0]['ground_truth'])
                    res['base_images'][0]['ground_truth'].append({
                                                                "pixel_position": obj["object_geographies"][0]["image_geometry"],
                                                                "orig_projection": obj["object_geographies"][0]["srs"],
                                                                "orig_wkt": obj["object_geographies"][0]["geometry"],
                                                                "wkt_epsg_4326": obj["object_geographies"][0]["geography"],
                                                                "class": obj["object_types"][0]["name"]
                                                                })

            result.append(res)


    with open(json_file_path, 'w') as fp:
        json.dump(result, fp, indent=4)

    return result

# my_req = "http://localhost:4000/api/v1/dataset/f73e8f1f-f23f-4dca-8090-a40c4e1c260e?include_nested=t"
# my_req = "http://localhost:4000/api/v1/base-image?geography=ST_INTERSECTS(ST_GEOMFROMTEXT(POLYGON%20((126.2228869226379%2045.61847917970996%2C%20126.233513403451%2045.61873720815048%2C%20126.2338803282899%2045.61128017104979%2C%20126.2232552524807%2045.61102220932946%2C%20126.2228869226379%2045.61847917970996))))"


# request = requests.get(my_req).json()
# print(len(request[0]))
# print(request)
# print(request[0]['tiles_path'])

dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
objects = remove_duplicates(dataset_id,read_json_file=False)
print(objects[0].keys())


# for req in request:
    # print(req.__dict__)
    # break
#   image_path_on_server = req['image_path']
#   # print(image_path_on_server)
#   image_path = image_path_on_server.replace("/datasets","/home/murat/Projects/airplane_recognition/DATA")

#   img = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
#   plt.imshow(img)
#   plt.show()