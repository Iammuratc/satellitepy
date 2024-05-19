import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import init_logger, get_project_folder, create_folder
from satellitepy.utils.mask_utils import find_random_connected_block, calculate_adjacent_coordinates, \
    is_valid_coordinates
from satellitepy.data.labels import read_label
import logging
import random
from PIL import Image, ImageDraw, ImageChops

project_folder = get_project_folder()
mask_color = (0, 0, 0)


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-path', type=Path, required=True,
                        help='Path to image that should be masked')
    parser.add_argument('--percentage', type=int, required=True,
                        help='What percentage of the image should be masked')
    parser.add_argument('--mode', type=str, required=True,
                        help='How the image should be masked (Random, Block, Grid)')
    parser.add_argument('--block-size', type=int, required=False, default=1,
                        help='Size of the Blocks used')
    parser.add_argument('--label-path', type=Path, required=False,
                        help='Path to label file')
    parser.add_argument('--label-format', type=str, required=False,
                        help='Label format for label file (satellitepy, dota, fair1m, etc.)')
    parser.add_argument('--skip-possibility', type=int, required=False, default=10,
                        help='Only used by block mode. Possibilty when a block should be skipped to greate "holes" (Default: 10)')
    parser.add_argument('--output-folder', type=Path, required=False,
                        default=project_folder / Path('docs/masked_images/'),
                        help='Folder where the masked image should be saved to. Default: satellitepy/docs/masked_images')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log file path.')
    args = parser.parse_args()
    return args


def run(args):
    image_path = Path(args.image_path)
    percentage = args.percentage
    mode = args.mode.lower()
    block_size = args.block_size
    output_folder = Path(args.output_folder)
    label_path = Path(args.label_path)
    label_format = args.label_format
    skip_possibility = args.skip_possibility

    assert create_folder(output_folder)

    log_path = output_folder / f'mask_image.log' if args.log_path == None else args.log_path
    logging.getLogger('PIL.TiffImagePlugin').setLevel(logging.CRITICAL + 1)
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger()
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')

    logger.info('Masking Image')

    label = read_label(label_path, label_format)
    image = Image.open(image_path)
    mask = Image.new("L", image.size, 0)

    bbox_format = "obboxes"
    if label[bbox_format][0] is None:
        bbox_format = 'hbboxes'

    if mode == 'fill':
        mask_bbox_fill(mask, label[bbox_format])
    elif mode == 'random':
        mask_bbox_random(mask, label[bbox_format], percentage, block_size)
    elif mode == 'block':
        mask_bbox_block(mask, label[bbox_format], percentage, block_size, skip_possibility)
    else:
        logger.error(f'Unknown mode: {mode}!')
        exit(1)

    masked_image = ImageChops.composite(Image.new('RGB', image.size, (0, 0, 0)), image, mask)
    masked_image.save(output_folder / Path(image_path.stem + '.png'))


def mask_bbox_fill(mask, bboxes):
    for points in bboxes:
        points = [(int(x), int(y)) for x, y in points]

        ImageDraw.Draw(mask).polygon(points, fill=255)


def mask_bbox_random(mask, bboxes, percentage, rectangle_side):
    for bbox in bboxes:
        points = [(int(x), int(y)) for x, y in bbox]

        xmin = min(points, key=lambda x: x[0])[0]
        ymin = min(points, key=lambda x: x[1])[1]
        xmax = max(points, key=lambda x: x[0])[0]
        ymax = max(points, key=lambda x: x[1])[1]

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        num_rectangles = int((percentage / 100) * ((width // rectangle_side) * (height // rectangle_side)))
        positions = [(x, y) for y in range(ymin, ymax + 1) for x in range(xmin, xmax + 1)]
        selected_positions = random.sample(positions, min(num_rectangles, len(positions)))

        for x, y in selected_positions:
            if point_inside_polygon(x, y, points):
                mask.paste(255, (x, y, x + rectangle_side, y + rectangle_side))


def mask_bbox_block(mask, bboxes, percentage, rectangle_side, skip_probability):
    for bbox in bboxes:
        points = [(int(x), int(y)) for x, y in bbox]

        xmin = min(points, key=lambda x: x[0])[0]
        ymin = min(points, key=lambda x: x[1])[1]
        xmax = max(points, key=lambda x: x[0])[0]
        ymax = max(points, key=lambda x: x[1])[1]

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        num_blocks = int((percentage / 100) * ((width * height) / (rectangle_side ** 2)))
        grid = [[False] * (width // rectangle_side) for _ in range(height // rectangle_side)]

        start_x = random.randint(0, len(grid[0]) - 1)
        start_y = random.randint(0, len(grid) - 1)
        grid[start_y][start_x] = True
        num_connected = 1

        while num_connected < num_blocks:
            connected_x, connected_y = find_random_connected_block(grid)

            direction = random.choice(['up', 'down', 'left', 'right'])

            new_x, new_y = calculate_adjacent_coordinates(connected_x, connected_y, direction)

            if is_valid_coordinates(new_x, new_y, len(grid[0]), len(grid)) and not grid[new_y][new_x]:
                if random.random() > skip_probability / 100:
                    grid[new_y][new_x] = True
                    num_connected += 1

                    start_x = new_x * rectangle_side
                    start_y = new_y * rectangle_side

                    for dx in range(rectangle_side):
                        for dy in range(rectangle_side):
                            pixel_x = xmin + start_x + dx
                            pixel_y = ymin + start_y + dy

                            if point_inside_polygon(pixel_x, pixel_y, points):
                                mask.paste(255, (pixel_x, pixel_y, pixel_x + 1, pixel_y + 1))

                else:
                    grid[new_y][new_x] = True


def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


if __name__ == '__main__':
    args = get_args()
    run(args)
