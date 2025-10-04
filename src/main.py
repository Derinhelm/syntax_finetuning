import argparse
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

print(configs)

parameters = Parameters(configs)

os.makedirs(parameters.output_dir_path)
with open(parameters.output_dir_path + config_name.split('/')[-1], 'w') as file:
    yaml.dump(configs, file, default_flow_style=False)

print("LEARNING_RATE:\t" + str(parameters.learning_rate))

conduct_experiment(parameters)