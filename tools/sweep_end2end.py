import logging
from pathlib import Path

from satellitepy.utils.path_utils import get_project_folder, create_folder, init_logger
from satellitepy.utils.wandb.wandb_utils import get_sweep_config, init_sweep, run_sweep_agent

default_config = {
    'data': {
        'train': {
            'image_folder': '/raid/userdata/j0nl0060/data/dotView1M_patches/combined/train/images/',
            'label_folder': '/raid/userdata/j0nl0060/data/dotView1M_patches/combined/train/labels/',
            'mask_folder:': None,    # Not implemented, use satpy labels
            'label_format': 'satellitepy'
        },
        'val': {
            'image_folder': '/raid/userdata/j0nl0060/data/dotView1M_patches/combined/val/images/',
            'label_folder': '/raid/userdata/j0nl0060/data/dotView1M_patches/combined/val/labels/',
            'mask_folder:': None,   # Not implemented, use satpy labels
            'label_format': 'satellitepy'
        },
        'test': {
            'image_folder': '/raid/userdata/j0nl0060/data/dotView1M_patches/combined/test/images/',
            'label_folder': '/raid/userdata/j0nl0060/data/dotView1M_patches/combined/test/labels/',
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
        'batch_size': 8
    },
    'model': {
        'resnet_type': 101,
        'down_ratio': 4,
        'tasks': ['hbboxes', 'obboxes', 'coarse-class', 'fine-class'],

        'SWEEP_target_task': {
            'values': ['coarse-class', 'fine-class', 'very-fine-class']
        },
        # 'target_task': 'coarse-class',

        'K': 500,
        'random_seed': 42424242,

        "SWEEP_init_lr": {
                        'distribution': 'uniform',
                        'min': 0.0001,
                        'max': 0.001
                },
        # 'init_lr': 1.25e-4,

        'num_epoch': 100,
        'num_workers': 4,
        'ngpus': 2,
        'conf_thresh': 0.1, # Not used??
        'checkpoint_path': None,
        'patience': 5
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
    'out_folder': '/mnt/2tb-0/satellitepy/temp/sweep_demo',
    'wandb': True,
    'wandb_project': 'dotView1M',
    'wandb_run_name': ''
}

if __name__ == '__main__':
    msg = f'Sweep Id; Leave empty to start a new sweep:'
    sweep_id = input(msg)

    project_folder = get_project_folder()
    out_folder = Path(default_config['out_folder'])
    assert create_folder(out_folder)

    log_path = Path(
        out_folder) / 'train_bbavector.log'
    init_logger(config_path=project_folder / Path('configs/log.config'), log_path=log_path)
    logger = logging.getLogger('')

    sweep_config = get_sweep_config(default_config)

    if not sweep_id:
        sweep_id = init_sweep(sweep_config, default_config['wandb_project'])

    run_sweep_agent(default_config, sweep_id, 10, default_config['wandb_project'], Path('./sweep_results'))