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

if isinstance(configs['dataset_config'], list):
    configs['dataset_config'] = [ DatasetConfig(path_c) for path_c in configs['dataset_config'] ]
else:
    configs['dataset_config'] = DatasetConfig(configs['dataset_config'])

if isinstance(configs['model_config'], list):
    configs['model_config'] = [ ModelConfig(path_c) for path_c in configs['model_config'] ]
else:
    configs['model_config'] = ModelConfig(configs['model_config'])

several_parameters = OrderedDict()
for param_name, param_values in configs.items():
    if isinstance(param_values, list):
        several_parameters[param_name] = param_values # Several parameters
    else:
        parameters.__setattr__(param_name, param_values) # One parameter

several_param_names = list(several_parameters.keys())
s_params = list(itertools.product(*several_parameters.values()))

#os.makedirs(parameters.root_output_dir_path)
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

    os.makedirs(cur_parameters.output_dir_path)
    conduct_experiment(cur_parameters)
print("Finish")
