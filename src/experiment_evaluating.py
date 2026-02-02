
import argparse
import os
import yaml

from conllu import parse
import json
import gc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Getting config name')
    parser.add_argument('-c','--config', help='Config Name',
                    default="score_config.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
    gold_path = config["gold_file"]
    
    format = config.get("format", "jsonl")
    metric_type = config.get("metric_type", "difference")

    if metric_type == "difference":
        from metric_functions.difference_tokens.difference_main import process_sentence, print_mean_metrics
        process_function = process_sentence
        print_function = print_mean_metrics
    elif metric_type == "normal":
        print("normal")
        1 / 0 # TODO

    pred_filenames = []
    for file_directory in config['pred_directories']:
        new_pred_filenames = os.listdir(file_directory)
        new_pred_filenames = [ f"{file_directory}/{filename}" for filename in new_pred_filenames if filename[0] != "." ]
        pred_filenames += new_pred_filenames

    
    with open(gold_path, 'r') as file:
        content = file.read()
    sentences = parse(content)
    
    
    config_uas, config_las = {}, {}
    for pred_filename in pred_filenames:
        print(pred_filename)
        try:
          if format == "jsonl":
            pred_trees = []
            with open(pred_filename, 'r', encoding='utf-8') as f:
              for line_num, line in enumerate(f, 1):         
                  item = json.loads(line)
                  pred_trees.append(item)
          else:
            with open(pred_filename, 'r', encoding='utf-8') as f:
                pred_trees = json.load(f)

          assert len(sentences) == len(pred_trees)
          config_uas[pred_filename], config_las[pred_filename] = [], []


          for sent_i, sent_r in enumerate(sentences):
            try:
                sent_uas, sent_las = process_function(sentences[sent_i], pred_trees[sent_i])
                config_uas[pred_filename].append(sent_uas)
                config_las[pred_filename].append(sent_las)
            except Exception as e:
                print(e)
                # TODO (для предложений из нескольких предложений)
                config_uas[pred_filename].append(1)
                config_las[pred_filename].append(1)

          del pred_trees
          gc.collect()
        except Exception as e:
          print(f"Error: {e}")
        
    for config_i, config_name in enumerate(config_uas):
        print(config_name)
        print_function(config_uas[config_name], config_las[config_name])
        print()
    #if (config_i + 1) % 4 == 0:
    #  print("-" * 15)
