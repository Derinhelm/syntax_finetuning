from src.metric_functions.metric_calculating import calculate_metrics
from src.metric_functions.category_calculating import create_statistics
    
import argparse
import os
import yaml

from conllu import parse
import json
import gc
    
def print_mean_metrics(uas_metrics, las_metrics):
    good_uas = [r for r in uas_metrics if r is not None]
    bad_uas = [r for r in uas_metrics if r is None]
    good_las = [r for r in las_metrics if r is not None]
    bad_las = [r for r in las_metrics if r is None]
    if good_uas:
        print(f"{sum(good_uas) / len(good_uas) * 100:.2f}% ({sum(good_uas) / len(uas_metrics) * 100:.2f}%)")
    if good_las:
        print(f"{sum(good_las) / len(good_las) * 100:.2f}% ({sum(good_las) / len(las_metrics) * 100:.2f}%)")
    print(len(bad_uas), len(uas_metrics))#len(bad_las))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Getting config name')
    parser.add_argument('-c','--config', help='Config Name',
                    default="score_config.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
    gold_path = config["gold_file"]

    pred_filenames = []
    for file_directory in config['pred_directories']:
        new_pred_filenames = os.listdir(file_directory)
        new_pred_filenames = [ f"{file_directory}/{filename}" for filename in new_pred_filenames if filename[0] != "." ]
        pred_filenames += new_pred_filenames

    
    with open(gold_path, 'r') as file:
        content = file.read()
    sentences = parse(content)
    
    
    config_results = {}
    config_uas, config_las = {}, {}
    config_errors = {}
    config_extra_pred_tokens = {}
    for pred_filename in pred_filenames:
        print(pred_filename)
        try:
          pred_trees = []
          with open(filepath, 'r', encoding='utf-8') as f:
              for line_num, line in enumerate(f, 1):         
                  item = json.loads(line)
                  pred_trees.append(item)

          assert len(sentences) == len(pred_trees)
          config_results[pred_filename] = []
          config_errors[pred_filename] = []
          config_extra_pred_tokens[pred_filename] = []
          for sent_i in range(len(sentences)):
            try:
                output_r, output_errors, output_extra_pred_tokens = create_statistics(sentences[sent_i], pred_trees[sent_i])
                config_results[pred_filename].append(output_r)
                config_extra_pred_tokens[pred_filename].append(output_extra_pred_tokens)
                if output_errors:
                    config_errors[pred_filename].append(output_errors)
            except Exception:
                config_results[pred_filename].append(None)
                # TODO (для предложений из нескольких предложений)

          config_uas[pred_filename], config_las[pred_filename] = [], []
          for sent_r in config_results[pred_filename]:
            if r is None: # TODO
                config_uas[pred_filename].append(1)
                config_las[pred_filename].append(1)
            else:   
              sent_uas, sent_las = \
                calculate_metrics(sent_r)
              config_uas[pred_filename].append(uas)
              config_las[pred_filename].append(las)
          del pred_trees
          gc.collect()
        except Exception as e:
          print(f"Error: {e}")
        
    for config_i, config_name in enumerate(config_uas):
        print(config_name)
        print_mean_metrics(config_uas[config_name], config_las[config_name])
        print()
    #if (config_i + 1) % 4 == 0:
    #  print("-" * 15)