import json

from metric_functions.evaluate_one import evaluate_one_experiment, calculate_mean_metrics

from conllu import parse

def conduct_evaluation(output_experiment_path, dataset_config, result_path, metric):
        res_name = "_".join(result_path.split("/")[-1].split(".")[:-1])
        metric_path = f"{output_experiment_path}/metrics_{res_name}.jsonl"
        conll_test_file_path = dataset_config.conll_test_file_path

        with open(conll_test_file_path, 'r') as file:
            content = file.read()
        gold_sentences = parse(content)

        metric_format = metric["format"]
        metric_type = metric["metric_type"]

        expir_res_uas, expir_res_las, expir_res_coeffs = evaluate_one_experiment(
            gold_sentences, result_path, metric_format, metric_type)
        
        short_filename = metric_path.split("/")[-1].split(".")[0]
        results = {}
        results[f"{short_filename}_mean"] = calculate_mean_metrics(expir_res_uas,
            expir_res_las, expir_res_coeffs)
        results[f"{short_filename}_uas"] = expir_res_uas
        results[f"{short_filename}_las"] = expir_res_las
        results[f"{short_filename}_coeffs"] = expir_res_coeffs
        with open(metric_path, 'w') as f:
            json.dump(results, f, indent=4)
