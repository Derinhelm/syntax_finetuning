import copy
import json
import os
import time

from conllu import parse


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


def get_pred_trees(pred_filename, pred_format):
    if pred_format == "jsonl":
        pred_trees = []
        with open(pred_filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):         
                item = json.loads(line)
                pred_trees.append(item)
    elif pred_format == "conll":
        with open(pred_filename, 'r') as file:
            content = file.read()
        conll_sentences = parse(content)
        pred_trees = []
        for i, conll_res in enumerate(conll_sentences):
            tree_nodes = [{'id': str(t['id']), 'form': t['form'],
                    'parent_id': str(t['head']), 'relation': t['deprel']}
                    for t in conll_res ]
            pred_trees.append({'index': i, 'pred_tree': tree_nodes})
    else:
        with open(pred_filename, 'r', encoding='utf-8') as f:
            pred_trees = json.load(f)
    return pred_trees

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
                            with open(conll_test_file_path, 'r') as file: # TODO: вынести из for ?
                                content = file.read()
                            gold_sentences = parse(content)

                            pred_format = metric["format"]
                            pred_trees = get_pred_trees(result_path, pred_format)
                            if len(gold_sentences) != len(pred_trees):
                                        print(f"Gold sents: {len(gold_sentences)}, pred sents: {len(pred_trees)}")
                                        gold_sent_ids = set(range(len(gold_sentences)))
                                        pred_sent_ids = {s['index'] for s in pred_trees}
                                        #print(list(gold_sent_ids)[:10])
                                        #print(list(pred_sent_ids)[:10])
                                        print(f"Extra: {sorted(list(pred_sent_ids - gold_sent_ids))}")
                                        print(f"Lost: {sorted(list(gold_sent_ids - pred_sent_ids))}")
                            
                            assert len(gold_sentences) == len(pred_trees)
                            res_name = "_".join(result_path.split("/")[-1].split(".")[:-1])

                            pred_format = metric["format"]
                            metric_type = metric["metric_type"]
                            metric_path = f"{parameters.output_experiment_path}/metrics_{res_name}_{pred_format}_{metric_type}.jsonl"

                            conduct_evaluation(gold_sentences, pred_trees, metric_path, metric)
                        except Exception as e:
                            print(f"Error during evaluation {result_path}:\n{e}")
    if not parameters.check_is_none():
        mark_ready(parameters.output_experiment_path)
