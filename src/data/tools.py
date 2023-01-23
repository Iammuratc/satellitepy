import json
import os


def normalize_rarePlanes_annotations(dataset_settings):
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
            label_file = open(label_file_path, 'r+')
            annotations = json.load(file)
            label_file.close()
            annotations['annotations'].append(new_annotation)
            file = open(label_file_path, 'w')

            json.dump(annotations, file, ensure_ascii=False, indent=4)


def get_file_paths(folder, sort=True):
    file_paths = [os.path.join(folder, file)
                  for file in os.listdir(folder)]
    if sort:
        file_paths.sort()
    return file_paths
