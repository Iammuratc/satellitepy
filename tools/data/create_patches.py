import configargparse

"""
Create patches from original image and label folders 
Save patch label details (i.e., corners, class_name) in json files in satellitepy format.
"""

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-size', required=True, type=int, help='Patch size. Patches with patch_size will be created from the original images.')
    parser.add_argument('--in-image-folder', required=True, help='Folder of original images. The images in this folder will be processed.')
    parser.add_argument('--in-label-folder', required=True, help='Folder of original labels. The labels in this folder will be used to create patches.')
    parser.add_argument('--in-label-format', required=True, help='Label file format. e.g., dota, fair1m, rareplanes.')
    parser.add_argument('--out-folder',
        required=True,
        help='Save folder of patches. Patches and corresponding labels will be saved into <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels.')
    parser.add_argument('--truncated-object-thr', default=0.5, type=float, help='If (truncated-object-thr x object area) is not in the patch area, the object will be filtered out.' 
        '1 if the object is completely in the patch, 0 if not.')
    parser.add_argument('--patch-overlap', required=True, type=int, help='Overlapping size of neighboring patches. In CNN terminology, stride = patch_size - patch_overlap')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
