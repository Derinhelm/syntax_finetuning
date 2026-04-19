import argparse
import os
import traceback
import yaml

from conllu import parse
import json

from metric_functions.evaluate_one import evaluate_one_experiment, calculate_mean_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Getting config name')
    parser.add_argument('-c','--config', help='Config Name',
                    default="score_config.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
    gold_path = config["gold_file"]
    
    pred_format = config.get("format", "jsonl")
    metric_type = config.get("metric_type", "difference")
    result_path = config.get("result_name", "result.json")

    pred_filenames = []
    for file_directory in config['pred_directories']:
        new_pred_filenames = os.listdir(file_directory)
        new_pred_filenames = [ f"{file_directory}/{filename}"
            for filename in new_pred_filenames if filename[0] != "." ]
        pred_filenames += new_pred_filenames
    
    with open(gold_path, 'r') as file:
        content = file.read()
    gold_sentences = parse(content)
    
    results = {}
    for pred_filename in sorted(pred_filenames):
        print(pred_filename)
        try:
            expir_res_uas, expir_res_las = evaluate_one_experiment(
                gold_sentences, pred_filename, pred_format, metric_type)
            mean_res = calculate_mean_metrics(expir_res_uas, expir_res_las)
            print(f"UAS: {mean_res['uas_right'] * 100:.2f}% ({mean_res['uas_all'] * 100:.2f}%), " +
                f"LAS: {mean_res['las_right'] * 100:.2f}% ({mean_res['las_all'] * 100:.2f}%)")
            print(mean_res['wrong_amount'], mean_res['all_amount'])#len(bad_las))
            short_filename = pred_filename.split("/")[-1].split(".")[0]
            results[f"{short_filename}_uas"] = expir_res_uas
            results[f"{short_filename}_las"] = expir_res_las
            results[f"{short_filename}_mean"] = mean_res

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.print_exc())

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4) # Using indent for pretty-printing
