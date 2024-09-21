import json
from copy import deepcopy
from functools import partial
from pathlib import Path
import time

import wandb

from tools.end2end import run_end2end

DEFAULT_SWEEP_CONFIG = {
    'method': 'random',
    'metric': {
        'name': 'target_mAP',
        'goal': 'maximize'
    },
    'parameters': { }
}

def get_sweep_config(
        original_config: dict,
        # to check other methods, go to https://docs.wandb.ai/guides/sweeps/sweep-config-keys#method
        method: str = DEFAULT_SWEEP_CONFIG['method'],
        metric_name: str = DEFAULT_SWEEP_CONFIG['metric']['name'],
        metric_goal: str = DEFAULT_SWEEP_CONFIG['metric']['goal']
) -> dict:
    sweep_config = DEFAULT_SWEEP_CONFIG
    sweep_config['method'] = method
    sweep_config['metric']['name'] = metric_name
    sweep_config['metric']['goal'] = metric_goal
    sweep_params = { }
    for param, values in get_all_sweep_params(original_config):
        sweep_params[param] = values

    if len(sweep_params) == 0:
        raise ValueError(f'No sweep params found in:\n{json.dumps(original_config, indent=4)}')

    sweep_config['parameters'] = sweep_params
    return sweep_config

def get_all_sweep_params(d):
    for key, value in d.items():
        if key.startswith('SWEEP_'):
            if not isinstance(value, dict):
                raise TypeError(f'Value of a sweep param {key} should be a dict with values / a distribution')
            yield key[6:], value

        elif isinstance(value, dict):
            yield from get_all_sweep_params(value)

def init_sweep(sweep_config: dict, wandb_project):
    sweep_id = wandb.sweep(sweep_config, project=wandb_project)
    print(f"Initialized sweep with id {sweep_id}")
    return sweep_id

def run_sweep_agent(original_config: dict, sweep_id: str, number_of_runs: int, sweep_project: str, save_path: Path):
    wandb.agent(sweep_id=sweep_id, function=partial(run_sweep_config, original_config, save_path, sweep_project), count=number_of_runs, project=sweep_project)


def run_sweep_config(original_config: dict, save_path: Path, sweep_project: str):
    with wandb.init(config=None, project=sweep_project, settings=wandb.Settings(start_method='spawn')) as wandb_run:
        # receive a config from a wandb agent (if this function is called by wandb.agent, as above)
        wandb_config = wandb.config

        new_config = deepcopy(original_config)
        fill_in_sweep_params(original_config, wandb_config, new_config)

        run_name = original_config['wandb_run_name']

        print(f"Running experiment {sweep_project}, run {run_name}\nResults will be saved to {save_path}\n")
        print(json.dumps(new_config, indent=4))

        run_end2end(new_config)

        if wandb_run:
            wandb_run.finish()

def fill_in_sweep_params(original_config, wandb_config: dict, new_config: dict):
    new_config['out_folder'] += '/' + time.strftime("%Y%m%d-%H%M%S")
    for key, value in original_config.items():
        if key.startswith('SWEEP_'):
            del new_config[key]
            new_config[key[6:]] = wandb_config[key[6:]]

        elif isinstance(value, dict):
            fill_in_sweep_params(value, wandb_config, new_config[key])