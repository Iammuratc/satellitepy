
"""
Test BBAVector models on original images.
Save detected bbox details (i.e., corners, class_name) with the corresponding ground truth labels.
For each available task, the matching score is calculated: mAP for classification tasks, relative score for regression tasks and IoU for masks.
"""

import configargparse
from satellitepy.evaluate.bbavector.tools import test_and_eval_original
from satellitepy.utils.path_utils import create_folder, init_logger
from pathlib import Path
import logging

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-path', help='Path to BBAVector model weights file.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--input-h', type=int, default=600, help='Resized image height. Equivalent of patch height.')
    parser.add_argument('--input-w', type=int, default=600, help='Resized image width. Equivalent of patch width.')
    parser.add_argument('--patch-size', type=int, default=600,
                        help='Patch size. Patches with patch_size will be created from the original image to be tested by the MMRotate model.'
                             'Be sure patch_size is the same as the input image size of your model, if not it will be resized to the input size.')
    parser.add_argument('--patch-overlap', type=int, default=100,
                        help='Overlapping size of neighboring patches. In CNN terminology, stride = patch_size - patch_overlap')
    parser.add_argument('--truncated-object-thresh', default=0.5, type=float,
                        help='If (truncated-object-thr x object area) is not in the patch area, the object will be filtered out. 1 if the object is completely in the patch, 0 if not.')
    parser.add_argument('--device', default='cuda:0', help='Device to load the model.')
    parser.add_argument('--tasks', default=['coarse-class'], nargs="+",
                        help='The tasks the model was trained on.'
                             'Find the other task names at satellitepy.data.utils.get_satellitepy_table.')
    parser.add_argument('--target-task', type=str, default='coarse-class',
                        help='The target task the model was trained on. Needs to be a classification task. Default is coarse-class')
    parser.add_argument('--log-config-path', default=Path("./configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--in-image-folder', help='Test image folder. The images in this folder will be tested.')
    parser.add_argument('--in-label-folder',
                        help='Test label folder. The labels in this folder will be used for evaluation purposes.')
    parser.add_argument('--in-label-format', default="satellitepy", help='Test label file format. e.g., dota, fair1m.')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask pixel coordinates in out labels.')
    parser.add_argument('--out-folder',
                        help='Save folder of detected bounding boxes. Predictions will be saved into <out-folder>.')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf-thresh', type=float, default=0.18,
                        help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--nms-iou-thresh', type=float, default=0.3,
                        help='Non-maximum suppression IOU threshold. Overlapping predictions will be removed according to this value.')

    parser.add_argument('--coarse-class-instance-names', type=str,
                        help='Instance names for CGC. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--fine-class-instance-names', type=str,
                        help='Instance names for FGC. The instance name --Background-- will be added automatically.  Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--very-fine-class-instance-names', type=str,
                        help='Instance names for FtGC. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--role-instance-names', type=str,
                        help='Instance names for Role. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--difficulty-instance-names', type=str,
                        help='Instance names for Difficulty. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--attributes_engines_no-engines-instance-names', type=str,
                        help='Instance names for attributes_engines_no-engines. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--attributes_engines_propulsion-instance-names', type=str,
                        help='Instance names for attributes_engines_propulsion. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--attributes_fuselage_canards-instance-names', type=str,
                        help='Instance names for attributes_fuselage_canards. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--attributes_wings_wing-shape-instance-names', type=str,
                        help='Instance names for attributes_wings_wing-shape. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--attributes_wings_wing-position-instance-names', type=str,
                        help='Instance names for attributes_wings_wing-position. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )
    parser.add_argument('--attributes_tail_no-tail-fins-instance-names', type=str,
                        help='Instance names for attributes_tail_no-tail-fins. The instance name --Background-- will be added automatically. Check tools/evaluate/default_instance_names.txt for default values.'
                        )

    parser.add_argument('--log-path', type=Path,
                        help='Log will be saved here. Default value is <out-folder>/evaluations.log')
    parser.add_argument('--ignore-other-instances', default=False, type=bool,
                        help='Ignores instances not in instance names if set. Default is False.')
    parser.add_argument('--mAP-confidence-score-thresholds', default=None, type=str,
                        help='Confidence score thresholds for evaluation of classification tasks. If the detected object has a lower'
                             'confidence score than this threshold, the object will be ignored. Default value is range(0,1.01,0.05).')
    parser.add_argument('--eval-iou-thresholds', default=None, type=str,
                        help='Iou thresholds. Default value is range(0.5,0.96,0.05)')
    parser.add_argument('--mask-confidence-score-threshold', default=0.5, type=str,
                        help='Confidence score threshold for evaluating masks and regression tasks. If the detected object has a lower'
                             'confidence score than this threshold, the object will be ignored. Default value is 0.5')
    parser.add_argument('--mask-threshold', type=float, default=10,
                        help='C for cv2.adaptiveThreshold. Value is subtracted from the threshold. Default is 10.')
    parser.add_argument('--mask-adaptive-size', type=float, default=101,
                        help='The threshold is the weighted sum of values in a neighbourhood of this size. Must be odd, default is 51.')
    args = parser.parse_args()
    return args



def main(args):
    """Application entry point."""
    log_config_path = Path(args.log_config_path)
    weights_path = args.weights_path
    device = args.device
    out_folder = Path(args.out_folder)
    in_image_folder = Path(args.in_image_folder)
    in_label_format = args.in_label_format
    tasks = args.tasks

    target_task = args.target_task
    assert target_task in tasks, "target task must be part of the tasks"

    conf_thresh = args.conf_thresh
    K = args.K
    input_h = args.input_h
    input_w = args.input_w
    num_workers = args.num_workers
    down_ratio = 4
    patch_overlap = args.patch_overlap
    patch_size = args.patch_size
    truncated_object_threshold = args.truncated_object_thresh
    nms_iou_threshold = args.nms_iou_thresh

    assert create_folder(out_folder)
    # Initiate logger

    log_path = Path(out_folder) / 'results.log' if args.log_path == None else args.log_path
    init_logger(config_path=log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info('BBAVector model will process the images...')
    in_label_folder = Path(args.in_label_folder)
    in_mask_folder = Path(args.in_mask_folder) if args.in_mask_folder else None

    instance_names = {}
    for task in tasks:
        if task in ['hbboxes', 'obboxes', 'masks', 'attributes_fuselage_length', 'attributes_wings_wing-span']:
            continue
        arg_name = task.replace('-', '_') + '_instance_names'
        task_instance_names = [name for name in args.__getattribute__(arg_name).split(',')] if args.__getattribute__(arg_name) else None
        instance_names[task] = task_instance_names

    ignore_other_instances = args.ignore_other_instances
    mAP_conf_score_thresholds = [float(confidence_score_threshold) for confidence_score_threshold in
                             args.mAP_confidence_score_thresholds.split(',')] if args.mAP_confidence_score_thresholds != None else [x / 100.0 for x in range(0, 96,5)]

    eval_iou_thresholds = [float(iou_threshold) for iou_threshold in
                      args.eval_iou_thresholds.split(',')] if args.eval_iou_thresholds != None else [x / 100.0 for x in range(50, 96, 5)]

    mask_conf_score_thersholds = float(args.mask_confidence_score_threshold)

    mask_threshold = int(args.mask_threshold) if args.mask_threshold else 10
    mask_adaptive_size = int(args.mask_adaptive_size) if args.mask_adaptive_size else 51
    assert mask_adaptive_size % 2 == 1, "mask-adaptive-size must be odd"

    test_and_eval_original(
        out_folder,
        in_image_folder,
        in_label_folder,
        in_mask_folder,
        in_label_format,
        weights_path,
        truncated_object_threshold,
        patch_size,
        patch_overlap,
        device,
        tasks,
        num_workers,
        input_h,
        input_w,
        conf_thresh,
        down_ratio,
        K,
        nms_iou_threshold,
        instance_names,
        mAP_conf_score_thresholds,
        eval_iou_thresholds,
        mask_conf_score_thersholds,
        mask_threshold,
        mask_adaptive_size,
        ignore_other_instances,

        target_task='coarse-class'
    )



if __name__ == '__main__':
    args = get_args()
    main(args)
