import torch

def torchify_satpy_label(value: list, possible_values: dict):
    if possible_values is not None:
        return torch.tensor([possible_values[v] if v is not None else torch.nan for v in value])
    else:
        return torch.tensor([v if v is not None else torch.nan for v in value])

def torchify_satpy_label_dict(satpy_label: dict, possible_values: dict):
    torchified = {}

    def inner(d: dict, pv: dict, prop_key = None):
        for k, values in d.items():
            inner_prop_key = k if prop_key == None else f"{prop_key}_{k}"
            if isinstance(values, dict):
                inner(values, pv[k], inner_prop_key)
            else:
                torchified[inner_prop_key] = torchify_satpy_label(values, pv[k])

    inner(satpy_label, possible_values)
    return torchified
