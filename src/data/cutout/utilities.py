import os

from PIL import Image

from src.data.cutout.cutout import Cutout


def filter_truncated_images(settings):
    """
    Filters out truncated images for every dataset part and 'cutout folder'
    by checking if the corners of an image are black
    Filtered images are saved in full_'cutout folder'
    """
    for dataset_part in settings['dataset_parts']:
        for folder in ['image_folder', 'orthogonal_image_folder', 'orthogonal_zoomed_image_folder']:

            src_folder = settings['cutout'][dataset_part][folder]

            folder_name = os.path.split(src_folder)[1]
            folder_path = os.path.split(src_folder)[0]

            full_images_folder = settings['cutout'][dataset_part]['full_' + folder]

            # truncated_images_folder = os.path.join(folder_path, folder_name + '_truncated_images')
            # create_folder(truncated_images_folder)

            img_paths = Cutout.get_file_paths('', src_folder, False)

            for img_path in img_paths:
                img_name = img_path.split('\\')[-1]
                img_name = img_name.split('/')[-1]
                print("Img_name: " + img_name)
                print(img_path)

                im = Image.open(img_path)
                px = im.load()
                width, height = im.size

                width_offset = width/3
                height_offset = height/3

                if ((px[0, 0] == (0, 0, 0) and px[width_offset, height_offset] == (0, 0, 0)) or
                        (px[0, height - 1] == (0, 0, 0) and px[width_offset, height - height_offset] == (0, 0, 0)) or
                        (px[width - 1, 0] == (0, 0, 0) and px[width - width_offset, height_offset] == (0, 0, 0)) or
                        (px[width - 1, height - 1] == (0, 0, 0) and px[width - width_offset, height - height_offset] == (0, 0, 0))):
                    print(f'Image {img_name} is truncated')
                else:
                    im.save(os.path.join(full_images_folder, img_name))