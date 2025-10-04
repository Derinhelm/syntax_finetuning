import argparse
import copy
import random
import os
import yaml

from deppllama_train_qlora import conduct_experiment
from parameters import Parameters, SeveralParameters

random.seed(23)

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", nargs='?', default='/src/src/configs/config.yaml')
parser_args = parser.parse_args()
config_name = parser_args.config_name

with open(config_name, 'r') as file:
    configs = yaml.safe_load(file)

print(configs)

main_parameters = Parameters(configs)
several_parameters = SeveralParameters(configs) # TODO: rename

os.makedirs(main_parameters.output_dir_path)
with open(main_parameters.output_dir_path + config_name.split('/')[-1], 'w') as file:
    yaml.dump(configs, file, default_flow_style=False)

print("LEARNING_RATE:\t" + str(main_parameters.learning_rate))

experiment_number = 0
for lora_alpha in several_parameters.lora_alpha: # TODO: автоматический перебор
    for lora_r in several_parameters.lora_r:
        for lora_dropout in several_parameters.lora_dropout:
            cur_parameters = copy.deepcopy(main_parameters)
            cur_parameters.output_dir_path = f"{cur_parameters.output_dir_path}/{experiment_number}"
            cur_parameters.lora_alpha = lora_alpha
            cur_parameters.lora_r = lora_r
            cur_parameters.lora_dropout = lora_dropout

            os.makedirs(cur_parameters.output_dir_path)
            conduct_experiment(cur_parameters)
            experiment_number += 1
