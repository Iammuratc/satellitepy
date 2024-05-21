import configargparse
import matplotlib.pyplot as plt
from pylab import imread
import json
import re


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--image-imd-file', type=str, required=False)
    parser.add_argument('--qgis-extent', type=str, required=False)
    parser.add_argument('--label-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()
    return args


def run(args):
    if not args.image_imd_file is None:
        with open(args.image_imd_file) as f:
            lines = f.readlines()

        ULX = get_float_out_of_file(lines, 'ULX')
        ULY = get_float_out_of_file(lines, 'ULY')
        URX = get_float_out_of_file(lines, 'URX')
        URY = get_float_out_of_file(lines, 'URY')
        LRX = get_float_out_of_file(lines, 'LRX')
        LRY = get_float_out_of_file(lines, 'LRY')
        LLX = get_float_out_of_file(lines, 'LLX')
        LLY = get_float_out_of_file(lines, 'LLY')

        y_min = min(ULY, URY, LRY, LLY)
        y_max = max(ULY, URY, LRY, LLY)
        x_min = min(ULX, URX, LRX, LLX)
        x_max = max(ULX, URX, LRX, LLX)

    elif args.qgis_extent is not None:
        coordinates = args.qgis_extent.replace(':', ',').replace(' ', '').split(',')
        y_min = float(min(coordinates[1], coordinates[3]))
        y_max = float(max(coordinates[1], coordinates[3]))
        x_min = float(min(coordinates[0], coordinates[2]))
        x_max = float(max(coordinates[0], coordinates[2]))

    else:
        print('Please provide either Qgis Extent Information or Path to IMD file')
        exit(1)

    im = imread(args.image_file)
    height = len(im)
    width = len(im[0])
    plt.imshow(im, extent=[0, width, 0, height])

    with open(args.label_file) as f:
        labels = json.load(f)

    for label in labels['features']:
        pixel_coordinates = []
        for coordinates in label['geometry']['coordinates'][0]:
            for coordinate in coordinates:
                pixel_coordinate = []
                x_pixel = round(width * (coordinate[0] - x_min) / (x_max - x_min))
                y_pixel = round(height * (coordinate[1] - y_min) / (y_max - y_min))
                pixel_coordinate.append(x_pixel)
                pixel_coordinate.append(y_pixel)
                pixel_coordinates.append(pixel_coordinate)
                plt.plot(x_pixel, y_pixel, marker='o', color='red')
        label['geometry']['coordinates'][0].append(pixel_coordinates)

    with open(args.output_file, 'w') as outfile:
        json.dump(labels, outfile)

    plt.show()


def get_float_out_of_file(lines, string):
    index = [i for i, s in enumerate(lines) if string in s]
    return float(re.findall('\d+\.\d+', lines[index[0]])[0])


if __name__ == '__main__':
    args = get_args()
    run(args)
