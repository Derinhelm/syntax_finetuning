import os
os.environ["VLLM_USE_V1"] = "0"
import argparse
import os
import yaml

from single_function_executor import InferenceExecutor
from start_experiments import run_all_experiments

from inference_parsing import create_inference_experiments

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

    inference_experiments = create_inference_experiments(configs)
    
    function_executor = InferenceExecutor()
    run_all_experiments(parallel_config, inference_experiments,
        function_executor)

if __name__ == "__main__":
    inference_main()
