import json
import os


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


def get_file_paths(folder, sort=True):
    file_paths = [os.path.join(folder, file)
                  for file in os.listdir(folder)]
    if sort:
        file_paths.sort()
    return file_paths
