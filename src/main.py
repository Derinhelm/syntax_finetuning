import argparse
import yaml

from inference_parsing import create_inference_experiments
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

    ft_experiments = create_finetuning_experiments(configs, config_name)
    print(f"FT experiment amount: {len(ft_experiments)}")

    inf_experiments = create_inference_experiments(configs)

    function_executor = FineTuningExecutor() # TODO: сделать выбор
    run_all_experiments(parallel_config, ft_experiments, inf_experiments,
            function_executor)

    print("Finish")

if __name__ == "__main__":
    finetuning_main()
