import argparse
import yaml


from single_function_executor import FineTuningExecutor
from start_experiments import run_all_experiments

from finetuning_parsing import create_finetuning_experiments

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

    experiments = create_finetuning_experiments(configs, config_name)

    print(f"Experiment amount: {len(experiments)}")

    function_executor = FineTuningExecutor() # TODO: сделать выбор
    run_all_experiments(parallel_config, experiments,
            function_executor)

    print("Finish")

if __name__ == "__main__":
    finetuning_main()
