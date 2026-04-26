import argparse
import copy
import os
import yaml

from parameters import Parameters

from single_function_executor import FineTuningExecutor
from start_experiments import run_all_experiments
from config_parsing import parse_field, get_several_config_params
from config import DatasetConfig, ModelConfig

def finetuning_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", nargs='?', default='/src/src/configs/config.yaml')
    parser_args = parser.parse_args()
    config_name = parser_args.config_name

    with open(config_name, 'r') as file:
        configs = yaml.safe_load(file)


    parallel_config = {}
    if "parallel_config" in configs:
        parallel_config = configs.pop("parallel_config")

    dataset_configs = parse_field(configs, "dataset_config", DatasetConfig)
    model_configs = parse_field(configs, "model_config", ModelConfig)

    parameters = Parameters(config_name)
    several_param_names, s_params = get_several_config_params(configs["finetuning"], parameters)

    experiments = []
    for model_i, model_config in enumerate(model_configs):
        for dataset_i, dataset_config in enumerate(dataset_configs):
            parameters.model_config = model_config 
            parameters.dataset_config = dataset_config
            os.makedirs(parameters.output_model_dataset_path, exist_ok=True)

            config_dir_path = parameters.config_dir_path
            with open(config_dir_path, 'w') as file:
                yaml.dump(configs, file, default_flow_style=False)

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

                os.makedirs(cur_parameters.output_experiment_path, exist_ok=True)
                experiments.append(cur_parameters)

    print(f"Experiment amount: {len(experiments)}")

    function_executor = FineTuningExecutor() # TODO: сделать выбор
    run_all_experiments(parallel_config, experiments,
            function_executor)

    print("Finish")

if __name__ == "__main__":
    finetuning_main()
