import argparse
from conllu import parse_tree, parse
import json
import gc
import random
import yaml

from src.sentence_utils import simplify_relations, tree2string_plain, \
    tree2string_loct, tree2string_lct, tree2string_grct

OP = '['
CP = ']'

def process_treebank_sample(input_files, output_path,
        representation, simple_relations):
  print("-" * 20)
  print(input_files)
  print(output_path)
  print(representation)
  print(simple_relations)
  res_list = []
  for input_file in input_files:
    with open(input_file, 'r') as file:
        content = file.read()

        trees = parse_tree(content)

        if simple_relations:
            for tree in trees:
                simplify_relations(tree)

        sentences = parse(content)

        for i in range(len(trees)):
            str_input = tree2string_plain(sentences[i])
                
            if representation == "conll" or representation == "conll_short":
                for token in sentences[i]:
                    for k in dict(token):
                        if k not in {"id", "form", "head", "deprel"}:
                            token[k] = "_"

                sentences[i].metadata = {}
                if representation == "conll":
                    sent_res = sentences[i].serialize().replace("\n\n", "\n")
                else:
                    sent_res = ""
                    for t_i, token in enumerate(sentences[i]):
                        t = sentences[i][t_i]
                        sent_res += f"{t['id']}\t{t['form']}\t{t['head']}\t{t['deprel']}\n"
            elif representation == "loct":
                sent_res = tree2string_loct(trees[i]).replace(" ", "")
            elif representation == "grct":
                sent_res = tree2string_grct(trees[i]).replace(" ", "")
            else:
                sent_res = tree2string_lct(trees[i]).replace(" ", "")
            res_list.append({"index": i, "input": str_input, "output": sent_res})
           
  print(output_path)
  with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(res_list, json_file, ensure_ascii=False, indent=4)
  del res_list
  for _ in range(3):
      gc.collect()
  
random.seed(23)

parser = argparse.ArgumentParser(description='Transform a dataset sample')
parser.add_argument('-c','--config', help='Config Name')
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
print(config)

simple_relations = config.get("simple_relations", False)
for representation in config["formats"]:
    for treebank_info in config["treebanks"]:
        input_directory = treebank_info["input_directory"]
        output_directory = treebank_info["output_directory"]
        prefix = treebank_info["prefix"]
        for sample in ["train", "dev", "test"]:
            input_files = [f"{input_directory}/{samp}"
                for samp in treebank_info[sample]]
            output_path = f"{output_directory}/{prefix}_{representation}_{sample}.json"
    
            process_treebank_sample(input_files, output_path,
                representation, simple_relations)

