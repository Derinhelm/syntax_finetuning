import json

from metric_functions.evaluate_one import evaluate_one_experiment, calculate_mean_metrics

from conllu import parse



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
