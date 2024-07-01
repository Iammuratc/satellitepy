import configargparse
import matplotlib.pyplot as plt
from pylab import imread
import json
import PIL.Image
import re
import rasterio

PIL.Image.MAX_IMAGE_PIXELS = 999999999

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-file', type=str, required=True)      
    parser.add_argument('--label-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()
    return args


def run(args):
    with open(args.label_file) as f:
        labels = json.load(f)

    with rasterio.open(args.image_file) as src:
        id = 0
        for label in labels["features"]:
            pixel_coordinates = []

            role = label["properties"]["Role"]

            if label["properties"]["Source"] == None and role == None:
                print("No source or role for label: ", label)
                exit(1) 

            if role != None:
                roles = ["Airliner", "Private Jet", "Propeller", "Propeller-Military", "Private Jet-Military", "Airliner-Military"]
                if role not in roles:
                    print("Invalid role for label: ", label)
                    exit(1)
            
            for coordinates in label["geometry"]["coordinates"][0]:
                for coordinate in coordinates:
                    pixel_coordinate = []
                    x_pixel, y_pixel = ~src.transform * (coordinate[0], coordinate[1])
                    pixel_coordinate.append(x_pixel)
                    pixel_coordinate.append(y_pixel)
                    pixel_coordinates.append(pixel_coordinate)
            label["geometry"]["coordinates"] = pixel_coordinates
            label["properties"]["id"] = id
            id += 1

    with open(args.output_file, 'w') as outfile:
        json.dump(labels, outfile)

if __name__ == '__main__':
    args = get_args()
    run(args)

