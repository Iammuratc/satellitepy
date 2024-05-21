from LibMTL.metrics import AccMetric, L1Metric
from LibMTL.loss import CELoss, MSELoss

TASK_TARGET_MAP = {
    'classes_0': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'classes_1': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'classes_2': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'attributes_engines_no-engines': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'attributes_engines_propulsion': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'attributes_fuselage_canards': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'attributes_fuselage_length': {
        'metrics': ['L1'],
        'metrics_fn': L1Metric(),
        'loss_fn': MSELoss(),
        'weight': [1]
    },
    'attributes_wings_wing-span': {
        'metrics': ['L1'],
        'metrics_fn': L1Metric(),
        'loss_fn': MSELoss(),
        'weight': [1]
    },
    'attributes_wings_wing-shape': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'attributes_wings_wing_position': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'attributes_tail_no-tail-fins': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
    'role': {
        'metrics': ['Acc'],
        'metrics_fn': AccMetric(),
        'loss_fn': CELoss(),
        'weight': [1]
    },
}


def build_targets(tasks_cfg: dict):
    targets = {}

    for k in tasks_cfg.keys():
        targets[k] = TASK_TARGET_MAP[k]

    return targets
