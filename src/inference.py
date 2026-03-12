import argparse
from datetime import datetime
import multiprocessing as mp
import os
import yaml

from inference_parser import start_inference_experiment, \
    start_parallel_inference_experiment, create_adapter_name

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
        index_finish = model_config.get('index_finish', None)
        assert not (index_set is not None and index_start is not None) # Не более одного ограничения
        assert not (index_set is not None and index_finish is not None) # Не более одного ограничения

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
                "index_finish": index_finish,
                "original_model_id": original_model_id,
                "peft_model_id": peft_model_id, "adapter_name": adapter_name,
                "output_dir": output_dir, "dataset_name": dataset_name,
                "dataset_repr": dataset_repr, "seed": seed,
                "dataset_path": dataset_path})

    if not parallel_config:
        for exp in experiments:
            start_inference_experiment(exp)
    else:
            process_num = 8
            exp_groups = [[] for _ in range(process_num)]
            for i, item in enumerate(experiments):
                exp_groups[i % process_num].append(item)

            mp.set_start_method('spawn', force=True)
             
            start_time = datetime.now().strftime("%D %H:%M:%S").replace("/", "_").replace(":", "_")
            processes = []
            parallel_path = parallel_config["parallel_path"]
            for i in range(process_num):
                p = mp.Process(target=start_parallel_inference_experiment,
                    args=(exp_groups[i], i, parallel_path, start_time))
                processes.append(p)
                p.start()
            
            # Ждем завершения всех процессов
            for p in processes:
                p.join()

if __name__ == "__main__":
    inference_main()
