import os
os.environ["VLLM_USE_V1"] = "0"
import argparse
import copy
import os
import yaml

from config_parsing import parse_field, get_several_config_params
from config import DatasetConfig, DataRestrictionConfig
from inference_parser import create_adapter_name
from parameters import InferenceParameters
from single_function_executor import InferenceExecutor
from start_experiments import run_all_experiments

def inference_main():
    os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "300"
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", nargs='?',
                        default='/src/src/configs/config.yaml')
    parser_args = parser.parse_args()
    config_name = parser_args.config_name

    with open(config_name, 'r') as file:
        configs = yaml.safe_load(file)
    print(configs, "-" * 20, sep="\n")

    parallel_config = configs.get("parallel_config", {})

    root_output_dir_path = configs['root_output_dir_path']

    dataset_configs = parse_field(configs, "dataset", DatasetConfig)

    parameters = InferenceParameters()
    several_param_names, s_params = get_several_config_params(
        configs["inference"], parameters)
    
    models = []
    for model_config in configs['models']:
        if "peft_model_id" in model_config:
            model_config['adapter_name'] = model_config['name']
            models.append(model_config)
        else:
            config_adapters = model_config['peft_group']
            peft_adapters = [(a, create_adapter_name(a)) for a in config_adapters]
            for peft_model_id, adapter_name in peft_adapters:
                adapter_model_dict = copy.deepcopy(model_config)
                adapter_model_dict['peft_model_id'] = peft_model_id
                adapter_model_dict['adapter_name'] = adapter_name
                models.append(adapter_model_dict)

    experiments = []
    for model_config in models:
        for dataset_i, dataset_config in enumerate(dataset_configs):
            for experiment_number, experiment_params in enumerate(s_params):
                assert len(experiment_params) == len(several_param_names)
                cur_parameters = copy.deepcopy(parameters)
                for param_i, param in enumerate(experiment_params):
                    cur_parameters.__setattr__(several_param_names[param_i], param)
                cur_parameters.experiment_number = experiment_number

                data_restriction_config = DataRestrictionConfig(model_config)
                experiments.append({"model_config": model_config,
                    "data_restriction_config": data_restriction_config,
                    "root_output_dir_path": root_output_dir_path,
                    "dataset_config": dataset_config,
                    "cur_parameters": cur_parameters})

    function_executor = InferenceExecutor() # TODO: сделать выбор
    run_all_experiments(parallel_config, experiments,
        function_executor)

if __name__ == "__main__":
    inference_main()
