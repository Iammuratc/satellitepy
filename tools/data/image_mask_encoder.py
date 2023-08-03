import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import init_logger, get_project_folder, create_folder
from satellitepy.utils.mask_utils import find_random_connected_block, calculate_adjacent_coordinates, is_valid_coordinates
import logging
import random
from PIL import Image

project_folder = get_project_folder()
mask_color=(0, 0, 0)

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-path', type=Path, required=True,
                        help='Path to image that should be masked')
    parser.add_argument('--percentage', type=int, required=True,
                        help='What percentage of the image should be masked')
    parser.add_argument('--mode', type=str, required=True,
                       help='How the image should be masked (Random, Block, Grid)' )
    parser.add_argument('--block-size', type=int, required=False, default=25,
                        help='Size of the Blocks used')
    parser.add_argument('--skip-possibility', type=int, required=False, default=10,
                        help='Only used by block mode. Possibilty when a block should be skipped to greate "holes" (Default: 10)')
    parser.add_argument('--output-folder', type=Path, required=False, default= project_folder / Path('docs/masked_images/'),
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
    skip_possibility = args.skip_possibility
    
    assert create_folder(output_folder)

    log_path = output_folder / f'mask_image.log' if args.log_path == None else args.log_path
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.CRITICAL + 1)
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    
    logger.info('Masking Image')

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = mask.load()
    
    if mode == "random":
        logger.info('Using random mode')
        mask_image_random(block_size, percentage, width, height, pixels)
    elif mode == "block":
        logger.info('Using block mode')
        mask_image_block(block_size, percentage, width, height, pixels, skip_possibility)
    else:
        logger.error(f"Unknown mode: {mode}!")
        exit(1)
    
    masked_image = Image.alpha_composite(image, mask)
    masked_image.save(output_folder / Path(image_path.stem + ".png"))
    logger.info('Finised Masking')

def mask_image_block(block_size, percentage, width, height, pixels, skip_possibility):
    num_blocks = int((percentage / 100) * ((width * height) / (block_size ** 2)))
    grid = [[False] * (width // block_size) for _ in range(height // block_size)]

    start_x = random.randint(0, len(grid[0]) - 1)
    start_y = random.randint(0, len(grid) - 1)
    grid[start_y][start_x] = True
    num_connected = 1

    while num_connected < num_blocks:
        connected_x, connected_y = find_random_connected_block(grid)

        direction = random.choice(['up', 'down', 'left', 'right'])

        new_x, new_y = calculate_adjacent_coordinates(connected_x, connected_y, direction)

        if is_valid_coordinates(new_x, new_y, len(grid[0]), len(grid)) and not grid[new_y][new_x]:
            if random.random() > skip_possibility/100:
                grid[new_y][new_x] = True
                num_connected += 1

                start_x = new_x * block_size
                start_y = new_y * block_size

                for dx in range(block_size):
                    for dy in range(block_size):
                        pixel_x = start_x + dx
                        pixel_y = start_y + dy

                        if (0 <= pixel_x < width and 0 <= pixel_y < height):
                            pixels[pixel_x, pixel_y] = mask_color
            else:
                grid[new_y][new_x] = True

def mask_image_random(block_size, percentage, width, height, pixels):
    num_blocks = int((percentage / 100) * ((width * height) / (block_size**2)))
    positions = []
    for x in range(0, width, block_size):
        for y in range(0, height, block_size):
            positions.append((x, y))

    random.shuffle(positions)

    for i in range(num_blocks):
        x, y = positions[i]
        for dx in range(block_size):
            for dy in range(block_size):
                new_x = x + dx
                new_y = y + dy
                if (0 <= new_x < width and 0 <= new_y < height):
                    pixels[new_x, new_y] = mask_color


if __name__ == '__main__':
    args = get_args()
    run(args)