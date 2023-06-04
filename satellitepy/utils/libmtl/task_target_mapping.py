from LibMTL.metrics import AccMetric
from LibMTL.loss import CELoss

TASK_TARGET_MAP = {
    'classes_0': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    }
}

def build_targets(tasks_cfg: dict):
    targets = {}

    for k in tasks_cfg.keys():
        targets[k] = TASK_TARGET_MAP[k]

    return targets
