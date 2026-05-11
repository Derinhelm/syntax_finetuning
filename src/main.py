import argparse
import copy
import yaml

from config_parsing import parse_field
from config import DatasetConfig, ModelConfig
from inference_parsing import create_inference_experiments
from inference_parser import create_adapter_name
from finetuning_parsing import create_finetuning_experiments
from single_function_executor import FineTuningExecutor
from start_experiments import run_all_experiments


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

    if "model_config" in configs:
        ft_model_configs = parse_field(configs, "model_config", ModelConfig)
        ft_models = {m.model_name: m for m in ft_model_configs}
    else:
        ft_models = {}

    dataset_configs = parse_field(configs, "dataset_config", DatasetConfig)
    assert len(set(dc.treebank for dc in dataset_configs)) == len(dataset_configs)
    # Все treebank различны
    datasets = {d.treebank: d for d in dataset_configs}

    ft_experiments = create_finetuning_experiments(configs, config_name,
        ft_models, datasets)
    if len(ft_experiments) == 1 and ft_experiments[0].check_is_none():
        print("Creating empty ft_parameters")
    else:
        print(f"FT experiment amount: {len(ft_experiments)}")

    inf_models = {}
    for model_config in configs.get('inference_models', []):
        model_name = model_config['name']
        if "peft_model_id" in model_config: # TODO: может не быть для слитного ft + inf
            model_config['adapter_name'] = create_adapter_name(model_name)
            inf_models[model_name] = model_config
        else:
            config_adapters = model_config['peft_group']
            peft_adapters = [(a, create_adapter_name(a))
                             for a in config_adapters]
            for peft_model_id, adapter_name in peft_adapters:
                adapter_model_dict = copy.deepcopy(model_config)
                adapter_model_dict['peft_model_id'] = peft_model_id
                adapter_model_dict['adapter_name'] = adapter_name
                inf_models[model_name] = adapter_model_dict

    inf_experiments = create_inference_experiments(configs,
        inf_models, datasets)
    if inf_experiments != []:
        print(f"INFERENCE experiment amount: {len(inf_experiments)}")

    function_executor = FineTuningExecutor() # TODO: сделать выбор
    run_all_experiments(parallel_config, ft_experiments, inf_experiments,
            function_executor)

    print("Finish")

if __name__ == "__main__":
    finetuning_main()
