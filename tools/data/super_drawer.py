from PIL import Image, ImageDraw
from satellitepy.utils.path_utils import get_project_folder
import sys

def main():
    
    label_file = open(get_project_folder() / "in_folder/DOTA/train/bounding_boxes/P0000.txt", "r")
    labels = label_file.read().split("\n")[2:-1]

    with Image.open(get_project_folder() / "in_folder/DOTA/train/images/P0000.png") as image:
            
        draw = ImageDraw.Draw(image)
        
        for label in labels:

            data = label.split(" ")
            cords = [int(float(cord)) for cord in data[:-2]]

            if data[-2] == "plane":
                draw.polygon([
                    (cords[0], cords[1]),
                    (cords[2], cords[3]),
                    (cords[4], cords[5]),
                    (cords[6], cords[7]),
                    ], 
                    outline = "RED",
                    width = 2)
    
    image.save(get_project_folder() / "in_folder/DOTA/train/P0000_LINED.png")        

if __name__ == "__main__":
    main()