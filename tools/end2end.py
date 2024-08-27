import logging
from pathlib import Path
import wandb

import torch
import json

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
            'image_folder': '/mnt/2tb-0/satellitepy/dotView1M/patches/xview/train/images',
            'label_folder': '/mnt/2tb-0/satellitepy/dotView1M/patches/xview/train/labels',
            'mask_folder:': None,    # Not implemented, use satpy labels
            'label_format': 'satellitepy'
        },
        'val': {
            'image_folder': '/mnt/2tb-0/satellitepy/dotView1M/patches/xview/val/images',
            'label_folder': '/mnt/2tb-0/satellitepy/dotView1M/patches/xview/val/labels',
            'mask_folder:': None,   # Not implemented, use satpy labels
            'label_format': 'satellitepy'
        },
        'test': {
            'image_folder': '/mnt/2tb-0/satellitepy/dotView1M/patches/xview/test/images',
            'label_folder': '/mnt/2tb-0/satellitepy/dotView1M/patches/xview/test/labels',
            'mask_folder': None,
            'label_format': 'satellitepy',
            'image_read_module': 'cv2',
        },
        'input_h': 600,
        'input_w': 600,
        'patch_size': 600,
        'patch_overlap': 100,
        'augmentation': None,
        'validate_datasets': False,
        'batch_size': 2
    },
    'model': {
        'resnet_type': 101,
        'down_ratio': 4,
        'tasks': ['hbboxes', 'obboxes', 'coarse-class', 'fine-class'],
        'target_task': 'coarse-class',
        'K': 500,
        'random_seed': 42424242,
        'init_lr': 1.25e-4,
        'num_epoch': 100,
        'num_workers': 1,
        'ngpus': 2,
        'conf_thresh': 0.1, # Not used??
        'checkpoint_path': None,
        'patience': 10
    },
    'testing': {
        'truncated_object_thresh': 0.5,
        'conf_thresh': 0.3,
    },
    'evaluation': {
        'nms_iou_threshold': 0.3,
        'plot_pr': False,
        'mask_conf_thresh': 0.5,
        'mask_threshold': 10,
        'mask_adaptive_size': 51
    },
    'out_folder': '/mnt/2tb-0/satellitepy/temp/test_xview_K',
    'wandb': False,
    'wandb_name': 'test'
}


def config_training(config, wandb_run=None):
    out_folder = Path(config['out_folder'])
    logger = logging.getLogger('')
    tasks = config['model']['tasks']
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
        target_task=target_task,
        wandb_run=wandb_run
    )
    ctrbox_obj.train_network()

def config_test(config, wandb_run=None):
    out_folder = Path(config['out_folder'])
    logger = logging.getLogger('')
    weights_path = out_folder / 'model_best.pth'

    result_folder = Path(out_folder) / 'results' / 'predictions'
    assert create_folder(result_folder, False)

    if 'masks' in config['model']['tasks']:
        mask_folder = Path(out_folder) / 'results' / 'masks'
        assert create_folder(mask_folder, False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    save_original_image_results(
        out_folder=out_folder,
        in_image_folder=Path(config['data']['test']['image_folder']),
        in_mask_folder=Path(config['data']['test']['mask_folder']) if config['data']['test']['mask_folder'] else None,
        in_label_folder=Path(config['data']['test']['label_folder']),
        in_label_format=config['data']['val']['label_format'],
        truncated_object_threshold=config['testing']['truncated_object_thresh'],
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
        down_ratio=config['model']['down_ratio'],
        target_task=config['model']['target_task'],
        img_read_module=config['data']['test']['image_read_module'])

def config_eval(config, wandb_run=None):
    out_folder = Path(config['out_folder'])
    logger = logging.getLogger('')
    result_folder = out_folder / 'results' / 'predictions'
    results = {}

    target_task = config['model']['target_task']

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
                task_out_folder,
                config['evaluation']['plot_pr'],
                config['evaluation']['nms_iou_threshold'],
                ignore_other_instances=True,
                no_probability=False,
                by_source=False,
                wandb_run=wandb_run
            )

            if task==target_task and wandb_run:
                wandb_run.log({'target_mAP': results[task][0]})


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
                target_task
            )
        else:
            logger.info(f'Evaluation for task {task} not implemented.')

    logger.info(results)
    if wandb_run:
        wandb_run.finish()

def run_end2end(config):
    project_folder = get_project_folder()
    out_folder = Path(config['out_folder'])

    with open(out_folder / 'config.json', 'w') as file:
        json.dump(config, file, indent=4)

    wandb_run = None
    if config['wandb']:
        wandb.login()
        wandb_run = wandb.init(project=config['wandb_run_name'], entity='satellitepy', config=config)
        wandb_run.save(out_folder / 'config.json')

    config_training(config, wandb_run)

    config_test(config, wandb_run)

    config_eval(config, wandb_run)

if __name__ == '__main__':
    project_folder = get_project_folder()
    out_folder = Path(default_config['out_folder'])
    assert create_folder(out_folder)

    log_path = Path(
        out_folder) / 'train_bbavector.log'
    init_logger(config_path=project_folder / Path('configs/log.config'), log_path=log_path)
    logger = logging.getLogger('')
    logger.info(f'The default log path will be used: {log_path}')

    run_end2end(default_config)

