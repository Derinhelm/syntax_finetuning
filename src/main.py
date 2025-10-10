import argparse
from collections import OrderedDict
import copy
import itertools
import random
import os
import yaml

from deppllama_train_qlora import conduct_experiment
from parameters import Parameters

random.seed(23)

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", nargs='?', default='/src/src/configs/config.yaml')
parser_args = parser.parse_args()
config_name = parser_args.config_name

with open(config_name, 'r') as file:
    configs = yaml.safe_load(file)

parameters = Parameters()
several_parameters = OrderedDict()
for param_name, param_values in configs.items():
    if isinstance(param_values, list):
        several_parameters[param_name] = param_values # One parameter
    else:
        parameters.__setattr__(param_name, param_values) # Several parameters

several_param_names = list(several_parameters.keys())
s_params = list(itertools.product(several_parameters.values()))

os.makedirs(parameters.root_output_dir_path)
with open(parameters.output_dir_path + config_name.split('/')[-1], 'w') as file:
    yaml.dump(configs, file, default_flow_style=False)

for experiment_number, experiment_params in enumerate(s_params):
    assert len(experiment_params[0]) == len(several_param_names)
    cur_parameters = copy.deepcopy(parameters)
    for param_i, param in enumerate(experiment_params[0]):
        cur_parameters.__setattr__(several_param_names[param_i], param)
    cur_parameters.experiment_number = experiment_number
    print("-" * 10, cur_parameters.__dict__, sep='\n')

    os.makedirs(cur_parameters.output_dir_path)
    conduct_experiment(cur_parameters)
