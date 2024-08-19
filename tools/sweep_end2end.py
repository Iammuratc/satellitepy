import logging
from pathlib import Path

from satellitepy.utils.path_utils import get_project_folder, create_folder, init_logger
from satellitepy.utils.wandb.wandb_utils import get_sweep_config, init_sweep, run_sweep_agent

default_config = {
    'data': {
        'train': {
            'image_folder': 'M:\miniFair1M\images',
            'label_folder': 'M:\miniFair1M\labels',
            'mask_folder:': None,    # Not implemented, use satpy labels
            'label_format': 'satellitepy'
        },
        'val': {
            'image_folder': 'M:\miniFair1M\images',
            'label_folder': 'M:\miniFair1M\labels',
            'mask_folder:': None,   # Not implemented, use satpy labels
            'label_format': 'satellitepy'
        },
        'test': {
            'image_folder': 'M:\miniFair1M\images',
            'label_folder': 'M:\miniFair1M\labels',
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
        'tasks': ['hbboxes', 'obboxes', 'coarse-class', 'role'],
        'SWEEP_target_task': {
            'values': ['coarse-class', 'role']
        },
        'K': 500,
        'random_seed': 42424242,
        "SWEEP_init_lr": {  # a distribution of possible values for a sweep
                        'distribution': 'uniform',
                        'min': 1.0e-4,
                        'max': 1.0e-3
                },
        'num_epoch': 5,
        'num_workers': 1,
        'ngpus': 0,
        'conf_thresh': 0.1, # Not used??
        'checkpoint_path': None,
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
    'out_folder': 'M:\miniFair1M\out',
    'wandb': True,
    'wandb_project': 'test',
    'wandb_run_name': 'test_name'
}

if __name__ == '__main__':
    msg = f'Sweep Id; Leave empty to start a new sweep:'
    sweep_id = input(msg)

    project_folder = get_project_folder()
    out_folder = Path(default_config['out_folder'])
    assert create_folder(out_folder)

    # log_path = Path(
    #     out_folder) / 'train_bbavector.log'
    log_path = Path('M:\log.log')
    init_logger(config_path=project_folder / Path('configs/log.config'), log_path=log_path)
    logger = logging.getLogger('')

    sweep_config = get_sweep_config(default_config)

    if not sweep_id:
        sweep_id = init_sweep(sweep_config, default_config['wandb_project'])

    run_sweep_agent(default_config, sweep_id, 10, default_config['wandb_project'], Path('./sweep_results'))