import os
os.environ["VLLM_USE_V1"] = "0"
import argparse
import os
import yaml

from inference_parser import start_inference_experiment, \
    create_adapter_name

from start_experiments import run_all_experiments
from start_process import start_parallel_inference_experiment



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

    dataset_config = configs['dataset']
    dataset_path = dataset_config['path']
    dataset_name = dataset_config['name']
    dataset_repr = dataset_config['representation_type']

    logit_parameters = configs.get("logit_parameters") # TODO: Сделать множественным параметром
    if logit_parameters is None:
        logit_parameters = [None]
    elif not isinstance(logit_parameters, list):
        logit_parameters = [logit_parameters]
    logit_parameters = [(lp if lp != 'None' else None) for lp in logit_parameters]

    experiments = []
    for model_config in configs['models']:
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
                "output_dir": output_dir, "dataset_name": dataset_name,
                "dataset_repr": dataset_repr, "seed": seed,
                "dataset_path": dataset_path,
                "logit_parameters": cur_logit_parameters})
            print(experiments[-1])
        print(len(experiments))

    run_all_experiments(parallel_config, experiments,
        start_inference_experiment, start_parallel_inference_experiment)

if __name__ == "__main__":
    inference_main()
