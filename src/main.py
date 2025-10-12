import argparse
from collections import OrderedDict
import copy
import itertools
import random
import os
import yaml

from config import DatasetConfig, ModelConfig
from deppllama_train_qlora import conduct_experiment
from parameters import Parameters

random.seed(23)

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", nargs='?', default='/src/src/configs/config.yaml')
parser_args = parser.parse_args()
config_name = parser_args.config_name

with open(config_name, 'r') as file:
    configs = yaml.safe_load(file)

parameters = Parameters(config_name)
dataset_configs, model_configs = [], []
several_parameters = OrderedDict()
for param_name, param_values in configs.items():
    if param_name == 'dataset_config':
        if isinstance(configs['dataset_config'], list):
            dataset_configs = [ DatasetConfig(path_c) for path_c in configs['dataset_config'] ]
        else:
            dataset_configs = [ DatasetConfig(configs['dataset_config']) ]
    elif param_name == 'model_config':
        if isinstance(configs['model_config'], list):
            model_configs = [ ModelConfig(path_c) for path_c in configs['model_config'] ]
        else:
            model_configs = [ ModelConfig(configs['model_config']) ]
    else:
        if isinstance(param_values, list):
            several_parameters[param_name] = param_values # Several parameters
        else:
            parameters.__setattr__(param_name, param_values) # One parameter

print(dataset_configs)
print(model_configs)
several_param_names = list(several_parameters.keys())
for model_config in model_configs:
    for dataset_config in dataset_configs:
        parameters.model_config = model_config 
        parameters.dataset_config = dataset_config
        os.makedirs(parameters.output_model_dataset_path)

        s_params = list(itertools.product(*several_parameters.values()))

        config_dir_path = parameters.config_dir_path
        with open(config_dir_path, 'w') as file:
            yaml.dump(configs, file, default_flow_style=False)

        for experiment_number, experiment_params in enumerate(s_params):
            print(experiment_params)
            print(several_param_names)
            assert len(experiment_params) == len(several_param_names)
            cur_parameters = copy.deepcopy(parameters)
            for param_i, param in enumerate(experiment_params):
                cur_parameters.__setattr__(several_param_names[param_i], param)
            cur_parameters.experiment_number = experiment_number
            print("-" * 10, cur_parameters.__dict__, sep='\n')

            os.makedirs(cur_parameters.output_experiment_path)
            conduct_experiment(cur_parameters)
print("Finish")
