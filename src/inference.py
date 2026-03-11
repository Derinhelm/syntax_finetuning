import argparse
import os
import yaml

from inference_parser import start_inference_experiment, create_adapter_name

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

    output_dir = configs['output_dir']
    seed = configs.get('seed', 42)

    dataset_config = configs['dataset']
    dataset_path = dataset_config['path']
    dataset_name = dataset_config['name']
    dataset_repr = dataset_config['representation_type']

    experiments = []
    for model_config in configs['models']:
        print(model_config)
        index_set = model_config.get('index_set', None)
        index_start = model_config.get('index_start', None)
        assert not (index_set is not None and index_start is not None) # Не более одного ограничения
        index_predicate = lambda ind: True
        if index_set is not None:
            index_predicate = lambda ind: ind in set(index_set)
        if index_start is not None:
            print(f"{index_start=}")
            index_predicate = lambda ind: ind >= index_start

        original_model_id = model_config['original_model_id']
        if "peft_model_id" in model_config:
            peft_adapters = [(model_config['peft_model_id'], model_config['name'])]
        else:
            config_adapters = model_config['peft_group']
            print(f"config_adapters:{config_adapters}")
            peft_adapters = [(a, create_adapter_name(a)) for a in config_adapters]
        for peft_model_id, adapter_name in peft_adapters:
            experiments.append({"model_config": model_config,
                "index_set": index_set, "index_start": index_start,
                "index_predicate": index_predicate,
                "original_model_id": original_model_id,
                "peft_model_id": peft_model_id, "adapter_name": adapter_name,
                "output_dir": output_dir, "dataset_name": dataset_name,
                "dataset_repr": dataset_repr, "seed": seed,
                "dataset_path": dataset_path})

    for exp in experiments:
        start_inference_experiment(exp)

if __name__ == "__main__":
    inference_main()
