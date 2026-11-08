import json
import gc

from conllu import parse

from metric_functions.evaluate_one import evaluate_one_experiment, calculate_mean_metrics


def conduct_evaluation(gold_sentences,
            pred_trees, metric_path, metric):
        metric_type = metric["metric_type"]
        expir_res_uas, expir_res_las, expir_res_coeffs = evaluate_one_experiment(
            gold_sentences, pred_trees, metric_type)
        
        short_filename = metric_path.split("/")[-1].split(".")[0]
        results = {}
        results[f"{short_filename}_mean"] = calculate_mean_metrics(expir_res_uas,
            expir_res_las, expir_res_coeffs)
        results[f"{short_filename}_uas"] = expir_res_uas
        results[f"{short_filename}_las"] = expir_res_las
        results[f"{short_filename}_coeffs"] = expir_res_coeffs
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

            del pred_trees
            gc.collect()
        except Exception as e:
            print(f"Error during evaluation {result_path}:\n{e}")
