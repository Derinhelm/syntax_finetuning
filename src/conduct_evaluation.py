import json
import gc

from conllu import parse

from metric_functions.data_preparing import preprocess_tree
from metric_functions.evaluate_one import evaluate_one_experiment


def calculate_count(metrics):
    return len(metrics)

def calculate_wrong_count(metrics):
    return sum(map(metrics, lambda x: x is None))

def calculate_mean_all(metrics):
    change_metrics = [r if r is not None else 0.0 for r in metrics]
    return sum(change_metrics) / len(change_metrics)

def calculate_mean_correct(metrics):
    change_metrics = [r for r in metrics if r is not None]
    return sum(change_metrics) / len(change_metrics)


def conduct_evaluation(data, metric_path, metric):
        metric_type = metric["metric_type"]

        expir_res_uas, expir_res_las = evaluate_one_experiment(data, metric_type)
        
        short_filename = metric_path.split("/")[-1].split(".")[0]
        results = {}
        mean_res = {}

        mean_res["uas_right"] = calculate_mean_correct(expir_res_uas)
        mean_res["uas_all"] = calculate_mean_all(expir_res_las)
        mean_res["las_right"] = calculate_mean_correct(expir_res_las)
        mean_res["las_all"] = calculate_mean_all(expir_res_las)
        mean_res["wrong_amount"] = calculate_wrong_count(expir_res_uas)
        mean_res["all_amount"] = calculate_count(expir_res_uas)
        results[f"{short_filename}_mean"] = mean_res
        
        results[f"{short_filename}_uas"] = expir_res_uas
        results[f"{short_filename}_las"] = expir_res_las
        with open(metric_path, 'w') as f:
            json.dump(results, f, indent=4)

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

def prepare_evaluation_data(conll_test_file_path, result_path, pred_format):
    with open(conll_test_file_path, 'r') as file: # TODO: вынести из for ?
        content = file.read()
    gold_sentences = parse(content)

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


    pred_trees_dict = {tree['index']:tree for tree in pred_trees}
    assert len(pred_trees_dict) == len(pred_trees)
    data = [] # (gold, pred)
    for sent_i, sent_r in enumerate(gold_sentences):
        try:
            if isinstance(pred_trees_dict[sent_i]["pred_tree"], list):
                gold_tree = [{'id': str(t['id']), 'form': t['form'],
                    'parent_id': str(t['head']), 'relation': t['deprel'],
                    'pos': t['upos'], 'feats': t['feats']}
                        for t in gold_sentences[sent_i]]
                gold_text = gold_sentences[sent_i].metadata['text']
                gold_tree = preprocess_tree(gold_tree)
                pred_tree = preprocess_tree(
                    pred_trees_dict[sent_i]["pred_tree"])
                data.append({"gold_text": gold_text, "gold_tree": gold_tree, "pred_tree": pred_tree})
            else:
                data.append({"gold_text": gold_text, "gold_tree": gold_tree, "pred_tree": None})
        except Exception as e:
                data.append({"gold_text": None, "gold_tree": None, "pred_tree": None})

    return data

def create_metric_path(parameters, result_path, metric):
    res_name = "_".join(result_path.split("/")[-1].split(".")[:-1])
    metric_type = metric["metric_type"]
    pred_format = metric["format"]
    metric_path = f"{parameters.output_experiment_path}/metrics_{res_name}_{pred_format}_{metric_type}.jsonl"
    return metric_path

def conduct_evaluation_main(parameters,
        inf_experiment, inf_result_path, metric): # TODO: rename

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
            data = prepare_evaluation_data(
                 conll_test_file_path, result_path, metric["format"])
            metric_path = create_metric_path(parameters, result_path, metric)

            conduct_evaluation(data, metric_path, metric)

            del data
            gc.collect()
        except Exception as e:
            print(f"Error during evaluation {result_path}:\n{e}")
