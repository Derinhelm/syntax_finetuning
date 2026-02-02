
import argparse
import os
import yaml

from conllu import parse
import json
import gc

from metric_functions.category_calculating \
            import create_statistics
from metric_functions.data_preparing import preprocess_tree


def get_pred_trees(pred_filename, format):
    if format == "jsonl":
        pred_trees = []
        with open(pred_filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):         
                item = json.loads(line)
                pred_trees.append(item)
    else:
        with open(pred_filename, 'r', encoding='utf-8') as f:
            pred_trees = json.load(f)
    return pred_trees

def print_mean_metrics(uas_metrics, las_metrics):
    good_uas = [r for r in uas_metrics if r is not None]
    bad_uas = [r for r in uas_metrics if r is None]
    good_las = [r for r in las_metrics if r is not None]
    bad_las = [r for r in las_metrics if r is None]
    print(f"UAS: {sum(good_uas) / len(good_uas) * 100:.2f}% ({sum(good_uas) / len(uas_metrics) * 100:.2f}%), " +
        f"LAS: {sum(good_las) / len(good_las) * 100:.2f}% ({sum(good_las) / len(las_metrics) * 100:.2f}%)")
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
    
    format = config.get("format", "jsonl")
    metric_type = config.get("metric_type", "difference")

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
            pred_trees = get_pred_trees(pred_filename, format)
            assert len(sentences) == len(pred_trees)
            config_uas[pred_filename], config_las[pred_filename] = [], []

            for sent_i, sent_r in enumerate(sentences):
                if isinstance(pred_trees[sent_i]["pred_tree"], list):
                    gold_tree = [{'id': str(t['id']), 'form': t['form'],
                        'parent_id': str(t['head']), 'relation': t['deprel'],
                        'pos': t['upos'], 'feats': t['feats']} for t in sentences[sent_i]]
                    gold_text = sentences[sent_i].metadata['text']
                    gold_tree = preprocess_tree(gold_tree)
                    pred_tree = preprocess_tree(pred_trees[sent_i]["pred_tree"])
                    sent_uas, sent_las = create_statistics(gold_text, gold_tree,
                        pred_tree, metric_type)
                else: # Предложение с некорректным результатом
                    sent_uas, sent_las = None, None
                config_uas[pred_filename].append(sent_uas)
                config_las[pred_filename].append(sent_las)

            del pred_trees
            gc.collect()

            print_mean_metrics(config_uas[pred_filename], config_las[pred_filename])
        except Exception as e:
            print(f"Error: {e}")
