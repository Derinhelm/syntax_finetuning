import copy
import os
import yaml

from parameters import Parameters

from config_parsing import parse_field, get_several_config_params
from config import DatasetConfig, ModelConfig


def create_finetuning_experiments(configs, config_name):
    dataset_configs = parse_field(configs, "dataset_config", DatasetConfig)
    datasets = {d.treebank: d for d in dataset_configs}

    model_configs = parse_field(configs, "model_config", ModelConfig)
    models = {m.model_name: m for m in model_configs}

    parameters = Parameters(config_name)
    several_param_names, s_params = get_several_config_params(
        configs["finetuning"], parameters)
    parameters.root_output_dir_path = configs['root_output_dir_path']

    for model_config in model_configs:
        for dataset_config in dataset_configs:
            parameters.model_config = model_config 
            parameters.dataset_config = dataset_config
            os.makedirs(parameters.output_model_dataset_path, exist_ok=True)

            config_dir_path = \
                parameters.root_output_dir_path + "/" + parameters.config_name.split('/')[-1]
            with open(config_dir_path, 'w') as file:
                yaml.dump(configs, file, default_flow_style=False)

    experiments = []
    for experiment_number, experiment_params in enumerate(s_params):
        #print(f"Model {model_i} from {len(model_configs)}")
        #print(f"Dataset {dataset_i} from {len(dataset_configs)}")
        #print(f"Experiment {experiment_number} from {len(s_params)}")
        #print(experiment_params)
        #print(several_param_names)
        assert len(experiment_params) == len(several_param_names)
        cur_parameters = copy.deepcopy(parameters)
        for param_i, param in enumerate(experiment_params):
            cur_parameters.__setattr__(several_param_names[param_i], param)
        cur_parameters.experiment_number = experiment_number
        #print("-" * 10, cur_parameters.__dict__, sep='\n')

        cur_parameters.model_config = \
            models[cur_parameters.model_parameters['model_name']] # TODO: Сделать класс

        cur_parameters.dataset_config = \
            datasets[cur_parameters.treebank_parameters['treebank_name']]
        os.makedirs(cur_parameters.output_experiment_path, exist_ok=True)
        experiments.append(cur_parameters)
    return experiments
