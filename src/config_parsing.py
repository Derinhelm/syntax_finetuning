from collections import OrderedDict
import itertools

from config import DatasetConfig, ModelConfig

def parse_datasets(configs):
    if "dataset_config" in configs:
        if isinstance(configs['dataset_config'], list):
            dataset_configs = [ DatasetConfig(path_c) for path_c in configs['dataset_config'] ]
        else:
            dataset_configs = [ DatasetConfig(configs['dataset_config']) ]
        configs.pop("dataset_config")
    else:
        dataset_configs = []
    print(dataset_configs)
    return dataset_configs

def parse_models(configs):
    if "model_config" in configs:
        if isinstance(configs['model_config'], list):
            model_configs = [ ModelConfig(path_c) for path_c in configs['model_config'] ]
        else:
            model_configs = [ ModelConfig(configs['model_config']) ]
        configs.pop("model_config")
    else:
        model_configs = []
    print(model_configs)
    return model_configs


def get_several_config_params(configs, parameters):
    several_parameters = OrderedDict()
    for param_name, param_values in configs.items():
        if isinstance(param_values, list):
            several_parameters[param_name] = param_values # Several parameters
        else:
            parameters.__setattr__(param_name, param_values) # One parameter

    several_param_names = list(several_parameters.keys())
    s_params = list(itertools.product(*several_parameters.values()))
    if not s_params:
        s_params = [{}]
    return several_param_names, s_params
