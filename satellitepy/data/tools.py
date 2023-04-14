import json
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from satellitepy.utils.path_utils import create_folder, get_file_paths, zip_matched_files
from satellitepy.data.patch import get_patches
from satellitepy.data.labels import read_label
from satellitepy.data.cutout.geometry import BBox

def save_patches(
    image_folder,
    label_folder,
    label_format,
    out_folder,
    truncated_object_thr,
    patch_size,
    patch_overlap,
    ):
    """
    Save patches from the original images
    Parameters
    ----------
    image_folder : Path
        Input image folder. Images in this folder will be processed.
    label_folder : Path
        Input label folder. Labels in this folder will be used to create patch labels.
    label_format : str
        Input label format.
    out_folder : Path
        Output folder. Patches and corresponding labels will be saved into <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels
    truncated_object_thr : float
        Truncated object threshold
    patch_size : int
        Patch size
    patch_overlap : int
        Patch overlap
    Returns
    -------
    Save patches in <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels
    """

    # Create output folders
    out_image_folder = Path(out_folder) / f'patch_{patch_size}' / 'images'
    out_label_folder = Path(out_folder) / f'patch_{patch_size}' / 'labels'

    assert create_folder(out_image_folder)
    assert create_folder(out_label_folder)

    for img_path, label_path in zip_matched_files(image_folder,label_folder):
        # Image
        img = cv2.imread(str(img_path))
        # Labels
        gt_labels = read_label(label_path,label_format)

        # Save results with the corresponding ground truth
        patches = get_patches(
            img,
            gt_labels,
            truncated_object_thr,
            patch_size,
            patch_overlap,
            )

        count_patches = len(patches['images'])
        for i in range(count_patches):
            # Get original image name for naming patch files
            img_name = img_path.stem

            # Patch starting coordinates
            patch_x0, patch_y0 = patches['start_coords'][i]

            # Save patch image
            patch_img = patches['images'][i]
            patch_image_path = Path(out_image_folder) / f"{img_name}_x_{patch_x0}_y_{patch_y0}.png" 
            cv2.imwrite(str(patch_image_path),patch_img)

            # Save patch labels
            patch_label = patches['labels'][i]
            patch_label_path = Path(out_label_folder) / f"{img_name}_x_{patch_x0}_y_{patch_y0}.json"
            with open(str(patch_label_path),'w') as f:
                json.dump(patch_label,f)


def show_labels_on_image(img_path,label_path,label_format,show_labels=True):
    # img_path = os.path.join(self.original_image_folder,f'{img_name}.tif')
    # print(img_path)
    # bbox_path = os.path.join(self.original_bbox_folder,f'{img_name}.xml')
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    gt_labels = read_label(label_path,label_format)

    # fig, ax = plt.subplots(1)
    fig = plt.figure(frameon=False)
    # fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    # ax.imshow(mask, alpha=0.5)

    for bbox in gt_labels['bboxes']:
        bbox_corners = np.array(bbox[:8]).astype(int).reshape(4, 2)
        BBox.plot_bbox(corners=bbox_corners, ax=ax, c='b', s=5, instance_name=None)
    plt.axis('off')
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    plt.show()
    return fig





def normalize_rarePlanes_annotations(dataset_settings):
    """
    rarePlanes uses one annotation file for all images in one dataset part.
    To work with the dataset more efficiently, this function creates a separate annotation file for every image.
    Apart from the extension, the annotation file has the same name as the matching image.
    The separate annotation files are saved in the 'bounding_boxes'-folder,
    the original annotation file needs to be placed in its root folder (dataset part)
    """
    for dataset_part in dataset_settings['dataset_parts']:
        original_base_folder = dataset_settings['original'][dataset_part]['base_folder']
        new_bbox_folder = dataset_settings['original'][dataset_part]['bounding_box_folder']

        annotation_path = get_file_paths(original_base_folder)[0]
        file = json.load(open(annotation_path, 'r'))

        id_to_img = {}
        for image in file['images']:
            id_to_img[image['id']] = image['file_name']
            label_file = open(os.path.join(new_bbox_folder, image['file_name'][:-3] + 'json'), 'w')
            annotations = {'annotations': []}
            json.dump(annotations, label_file, indent=4)
            label_file.close()

        for new_annotation in file['annotations']:
            img_name = id_to_img[new_annotation['image_id']]
            label_file_path = os.path.join(new_bbox_folder, img_name[:-3] + 'json')
            label_file = open(label_file_path, 'r', encoding="utf-8")
            print(label_file_path)
            annotations = json.load(label_file)
            label_file.close()
            annotations['annotations'].append(new_annotation)
            file = open(label_file_path, 'w', encoding="utf-8")

            json.dump(annotations, file, ensure_ascii=False, indent=4)
            file.close()
