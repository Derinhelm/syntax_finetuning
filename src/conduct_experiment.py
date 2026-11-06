import copy
import os
import time

def mark_ready(directory):
    """Помечает директорию как готовую, создавая файл READY"""
    ready_file = os.path.join(directory, "READY")
    with open(ready_file, 'w') as f:
        f.write(f"Completed at {time.ctime()}\n")

def is_ready(directory):
    """Проверяет, готова ли директория"""
    return os.path.exists(os.path.join(directory, "READY"))

def create_inference_config_by_finetuning(parameters, inf_exp_param):
    from inference_parser import create_adapter_name

    inf_exp = copy.deepcopy(inf_exp_param)
    # TODO: assert, если поля не пустые
    # TODO: assert, если не совпадают dataset
    inf_exp["model_config"].original_model_id = parameters.model_config.model_name
    inf_exp["model_config"].peft_model_id = parameters.output_experiment_path
    inf_exp["model_config"].is_instruct = parameters.model_config.is_instruct
    inf_exp["model_config"].representation_type_result = \
        parameters.dataset_config.treebank_repr # TODO: Согласовать dataset
    inf_exp["model_config"].adapter_name = \
        create_adapter_name(inf_exp["model_config"].original_model_id)

    inf_exp["cur_parameters"].model_name = parameters.model_config.model_name

    inf_exp["root_output_dir_path"] = parameters.output_experiment_path

    inf_exp['dataset_config'] = parameters.dataset_config
    return inf_exp

def conduct_experiment(parameters, inf_experiments, metric_list):
    if not parameters.check_is_none():
        if not is_ready(parameters.output_experiment_path):
            from src.deppllama_train_qlora import conduct_finetuning_experiment
            conduct_finetuning_experiment(parameters)
    # Нужна оптимизация для "раскладывания по процессам", если только Inference

    if inf_experiments != []:
            os.environ["VLLM_USE_V1"] = "0"
            for inf_exp in inf_experiments:
                if not parameters.check_is_none() and \
                        not inf_exp["cur_parameters"].stub_parameter:
                    inf_experiment = create_inference_config_by_finetuning(
                        parameters, inf_exp)
                else:
                    inf_experiment = inf_exp
                if not inf_exp["cur_parameters"].stub_parameter:
                    from inference_parser import start_inference_experiment
                    inf_result_path = start_inference_experiment(inf_experiment)
                else:
                    inf_result_path = None
                for metric in metric_list:
                    if "result_path" in metric:
                        result_path = metric["result_path"]
                    elif inf_result_path is not None:
                        result_path = inf_result_path
                    else:
                        print(f"Error. No result path for metric: {metric}")
                        result_path = None
                    if "conll_test_file_path" in metric:
                        conll_test_file_path = metric["conll_test_file_path"]
                    else: # TODO: убрать необходимость в datasets для metric
                        conll_test_file_path = inf_experiment['dataset_config'].conll_test_file_path
                    if result_path is not None:
                        try:
                            from conduct_evaluation import conduct_evaluation
                            conduct_evaluation(parameters.output_experiment_path,
                                conll_test_file_path, result_path, metric)
                        except Exception as e:
                            print(f"Error during evaluation {result_path}:\n{e}")
    if not parameters.check_is_none():
        mark_ready(parameters.output_experiment_path)
