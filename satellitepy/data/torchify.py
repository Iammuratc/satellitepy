import torch

from satellitepy.data.utils import get_satellitepy_table

def torchify_satpy_label(value: list, possible_values: dict):
    '''
    Transforms the values from annotations in satellitepy format to stackable tensors.

    Parameters
    ----------
    value: list
        The values
    possible_values: dict
        A dict corresponding to the possible values / transformation. Used to transform the value. Ruleset:
            0. any None value is transformed to torch.nan (works only for float tensors)
            1. if possible value is None, the values are taken as is (except for None values - see 0.)
            2. any classification-like value is transformed to the index given by the possible_values dict
            3. if the possible_value contains a "max" and "min" key, the values are expected to be floats
               that shall be normalized according to "max" and "min"
    '''
    if possible_values is not None:
        result = []

        for v in value:
            if v is None:
                result.append(torch.nan)
            elif v in possible_values:
                result.append(possible_values[v])
            elif "max" in possible_values and "min" in possible_values:
                max = possible_values["max"]
                min = possible_values["min"]
                normalized = (v - min) / (max - min)
                result.append(normalized)

        return torch.tensor(result)
    else:
        return torch.tensor([v if v is not None else torch.nan for v in value])

def torchify_satpy_label_dict(satpy_label: dict, possible_values: dict = get_satellitepy_table()):
    '''
    Transforms annotations in satellitepy format to a flattened disctionary of stackable tensors.
    '''
    torchified = {}

    def inner(d: dict, pv: dict, prop_key = None):
        for k, values in d.items():
            inner_prop_key = k if prop_key == None else f"{prop_key}_{k}"
            if isinstance(values, dict):
                inner(values, pv[k] if pv is not None and k in pv else None, inner_prop_key)
            else:
                torchified[inner_prop_key] = torchify_satpy_label(values, pv[k] if pv is not None and k in pv else None)

    inner(satpy_label, possible_values)
    return torchified
