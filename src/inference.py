import os
os.environ["VLLM_USE_V1"] = "0"
import argparse
import os
import yaml

from config_parsing import parse_field
from inference_parser import create_adapter_name
from single_function_executor import InferenceExecutor
from start_experiments import run_all_experiments
from config import DatasetConfig

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

    output_dir = configs['output_dir']
    seeds = configs.get('seed', 42)
    if not isinstance(seeds, list):
        seeds = [seeds]

    dataset_configs = parse_field(configs, "dataset", DatasetConfig)

    logit_parameters = configs.get("logit_parameters") # TODO: Сделать множественным параметром
    if logit_parameters is None:
        logit_parameters = [None]
    elif not isinstance(logit_parameters, list):
        logit_parameters = [logit_parameters]
    logit_parameters = [(lp if lp != 'None' else None) for lp in logit_parameters]

    experiments = []
    for model_config in configs['models']:
        for dataset_i, dataset_config in enumerate(dataset_configs):
            for cur_logit_parameters in logit_parameters:
                for seed in seeds:
                    #print(model_config)
                    index_set = model_config.get('index_set', None)
                    index_start = model_config.get('index_start', None)
                    index_finish = model_config.get('index_finish', None)
                    assert not (index_set is not None and index_start is not None) # Не более одного ограничения
                    assert not (index_set is not None and index_finish is not None) # Не более одного ограничения

                    original_model_id = model_config['original_model_id']
                    if "peft_model_id" in model_config:
                        peft_adapters = [(model_config['peft_model_id'], model_config['name'])]
                    else:
                        config_adapters = model_config['peft_group']
                        #print(f"config_adapters:{config_adapters}")
                        peft_adapters = [(a, create_adapter_name(a)) for a in config_adapters]
                    for peft_model_id, adapter_name in peft_adapters:
                        experiments.append({"model_config": model_config,
                            "index_set": index_set, "index_start": index_start,
                            "index_finish": index_finish,
                            "original_model_id": original_model_id,
                            "peft_model_id": peft_model_id, "adapter_name": adapter_name,
                            "output_dir": output_dir,
                            "seed": seed,
                            "dataset_config": dataset_config,
                            "logit_parameters": cur_logit_parameters})
                        print(experiments[-1])
                    print(len(experiments))

    function_executor = InferenceExecutor() # TODO: сделать выбор
    run_all_experiments(parallel_config, experiments,
        function_executor)

if __name__ == "__main__":
    inference_main()
