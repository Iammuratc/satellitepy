import configargparse
import pathlib
from pathlib import Path
from tqdm import tqdm
from satellitepy.utils.path_utils import get_file_paths
import hashlib
import numpy as np

def get_args():
    """Arguments parser."""
    parser.add_argument('--in-image-folder', type=Path,
                        help='Folder of original images. The images in this folder will be processed.')
    parser.add_argument('--in-label-folder', type=Path, required=False,
                        help='Folder of original labels. The labels in this folder will be used to create patches.')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Save folder of validated image and labels.')
    args = parser.parse_args()
    return args


def validate_dataset(args):
    in_image_folder = Path(args.in_image_folder)
    in_label_folder = Path(args.in_label_folder)
    out_folder = Path(args.out_folder)

    total = len(get_file_paths(in_image_folder))
    removed = 0
    pbar = tqdm(zip_matched_files(in_image_folder,in_label_folder), total=total, desc="validating data")

    for img_path, label_path in pbar:
        hash_str = str(img_path) + str(label_path) + str(self.random_seed)
        hash_bytes = hashlib.sha256(bytes(hash_str, "utf-8")).digest()[:4]
        np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
        image = cv2.imread(img_path.absolute().as_posix())
        labels = read_label(label_path,in_label_format)
        image_h, image_w, c = image.shape
        annotation = self.preapare_annotations(labels, image_w, image_h)#, img_path)
        image, annotation = self.utils.data_transform(image, annotation, self.augmentation)

        if self.all_tasks_available(annotation):
            self.items.append((img_path, label_path, in_label_format))
        else:
            removed += 1
            pbar.set_description(f"validating data (removed: {removed})")
