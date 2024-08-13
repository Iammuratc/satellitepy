import logging
from pathlib import Path

from satellitepy.data.utils import get_satellitepy_table
from satellitepy.dataset.bbavector.dataset_bbavector import BBAVectorDataset
from satellitepy.evaluate.bbavector.tools import save_original_image_results
from satellitepy.evaluate.tools import calculate_map, calculate_iou_score
from satellitepy.models.bbavector import train_model
from satellitepy.models.bbavector.utils import get_model
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder

default_config = {
    'data': {
        'train': {
            'image_folder': ...,
            'label_folder': ...,
            'mask_folder:': ...,    # Not implemented, use satpy labels
            'label_format': ...
        },
        'val': {
            'image_folder': ...,
            'label_folder': ...,
            'mask_folder:': ...,   # Not implemented, use satpy labels
            'label_format': ...
        },
        'test': {
            'image_folder': ...,
            'label_folder': ...,
            'mask_folder:': ...,
            'label_format': ...,
            'image_read_module': 'cv2',
        },
        'input_h': ...,
        'input_w': ...,
        'patch_size': ...,
        'patch_overlap:': ...,
        'augmentation': ...,
        'validate_datasets': ...,
        'batch_size': ...
    },
    'model': {
        'resnet_type': 101,
        'down_ratio': 4,
        'tasks': ['hbboxes', 'obboxes', 'coarse-class'],
        'target_task': 'coarse-class',
        'K': 500,
        'random_seed': ...,
        'init_lr': ...,
        'num_epoch': ...,
        'num_workers': ...,
        'ngpus': ...,
        'conf_thresh': ..., # Not used??
        'checkpoint_path': ...,
        'patience': 10
    },
    'testing': {
        'truncated_object_thresh': 0.5,
        'conf_thresh': 0.25,
    },
    'evaluation': {
        'nms_iou_threshold': 0.3,
        'plot_pr': False,
        'mask_conf_thresh': 0.5,
        'mask_threshold': 10,
        'mask_adaptive_size': 51
    },
    'out_folder': ...
}


def config_training(config):
    out_folder = Path(config['out_folder'])
    logger = logging.getLogger('')
    logger.info(f'The default log path will be used: {log_path}')
    tasks = config['model']['tasks'],
    target_task = config['model']['target_task']
    assert 'obboxes' in tasks or 'hbboxes' in tasks, 'Tasks must contain at least one type of bounding boxes.'
    assert target_task in tasks, 'target task must be part of the tasks'
    logger.info('Initiating the training of the BBAVector model...')
    train_dataset = BBAVectorDataset(
        Path(config['data']['train']['image_folder']),
        Path(config['data']['train']['label_folder']),
        config['data']['train']['label_format'],
        tasks,
        config['data']['input_h'],
        config['data']['input_w'],
        config['model']['down_ratio'],
        target_task,
        config['data']['augmentation'],
        config['data']['validate_datasets'],
        K=config['model']['K'],
        random_seed=config['model']['random_seed']
    )
    valid_dataset = BBAVectorDataset(
        Path(config['data']['val']['image_folder']),
        Path(config['data']['val']['label_folder']),
        config['data']['val']['label_format'],
        tasks,
        config['data']['input_h'],
        config['data']['input_w'],
        config['model']['down_ratio'],
        target_task,
        config['data']['augmentation'],
        config['data']['validate_datasets'],
        K=config['model']['K'],
        random_seed=config['model']['random_seed']
    )
    model = get_model(config['model']['tasks'], config['model']['down_ratio'], config['model']['resnet_type'])
    ctrbox_obj = train_model.TrainModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        model=model,
        tasks=tasks,
        down_ratio=config['model']['down_ratio'],
        out_folder=out_folder,
        init_lr=config['model']['init_lr'],
        num_epoch=config['model']['num_epoch'],
        batch_size=config['data']['batch_size'],
        num_workers=config['model']['num_workers'],
        conf_thresh=config['model']['conf_thresh'],  # Not used?
        ngpus=config['model']['ngpus'],
        resume_train=config['model']['checkpoint_path'],
        patience=config['model']['patience'],
        target_task=target_task
    )
    ctrbox_obj.train_network()

def config_test(config):
    out_folder = Path(config['out_folder'])
    weights_path = out_folder / 'model_best.pth'

    result_folder = Path(out_folder) / 'results' / 'predictions'
    assert create_folder(result_folder, False)

    if 'masks' in config['model']['tasks']:
        mask_folder = Path(out_folder) / 'results' / 'masks'
        assert create_folder(mask_folder, False)

    device = 'cuda:0'

    save_original_image_results(
        out_folder=out_folder,
        in_image_folder=config['data']['test']['image_folder'],
        in_mask_folder=config['data']['test']['mask_folder'],
        in_label_folder=config['data']['test']['label_folder'],
        in_label_format=config['data']['val']['label_format'],
        truncated_object_threshold=config['testing']['truncated-object-thresh'],
        patch_size=config['data']['patch_size'],
        patch_overlap=config['data']['patch_overlap'],
        checkpoint_path=weights_path,
        device=device,
        tasks=config['model']['tasks'],
        K=config['model']['K'],
        conf_thresh=config['testing']['conf_thresh'],
        num_workers=config['model']['num_workers'],
        input_h=config['data']['input_h'],
        input_w=config['data']['input_w'],
        down_ratio=['model']['down_ratio'],
        target_task=config['model']['target_task'],
        img_read_module=config['data']['test']['image_read_module'])

def config_eval(config):
    out_folder = Path(config['out_folder'])
    result_folder = out_folder / 'results' / 'predictions'
    results = {}

    for task in config['model']['tasks']:
        if task in ['obboxes', 'hbboxes']:
            continue
        task_out_folder = out_folder / 'results' / task

        assert create_folder(task_out_folder, False)

        if task in ['coarse-class', 'fine-class', 'very-fine-class', 'role']:
            instance_dict = get_satellitepy_table()[task]
            conf_score_thresholds = [x / 100.0 for x in range(0, 96, 5)]
            iou_thresholds = [x / 100.0 for x in range(50, 96, 5)]

            results[task] = calculate_map(
                result_folder,
                task,
                instance_dict,
                conf_score_thresholds,
                iou_thresholds,
                out_folder,
                config['evaluation']['plot_pr'],
                config['evaluation']['nms_iou_threshold'],
                ignore_other_instances=True,
                no_probability=False,
                by_source=False
            )
        elif task in ['length', 'wing-span']:
            logger.info(f'Evaluation for regression tasks not implemented yet.')
        elif task == 'masks':
            iou_thresholds = [x / 100.0 for x in range(50, 96, 5)]

            results[task] = calculate_iou_score(
                result_folder,
                config['data']['test']['mask_folder'],
                task_out_folder,
                iou_thresholds,
                config['evaluation']['mask_conf_thresh'],
                config['evaluation']['nms_iou_threshold'],
                config['evaluation']['mask_threshold'],
                config['evaluation']['mask_adaptive_size'],
                config['model']['target_task']
            )
        else:
            logger.info(f'Evaluation for task {task} not implemented.')

    logger.info(results)

if __name__ == '__main__':
    project_folder = get_project_folder()
    out_folder = Path(default_config['out_folder'])
    assert create_folder(out_folder)

    log_path = Path(
        out_folder) / 'train_bbavector.log'
    init_logger(config_path=project_folder / Path('configs/log.config'), log_path=log_path)
    logger = logging.getLogger('')

    config_training(default_config)

    config_test(default_config)

    config_eval(default_config)
