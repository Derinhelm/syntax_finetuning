import argparse
import gc
import json
import time
import yaml

import torch

from inference_functions.inferencer import LLMInferencer
from inference_functions.tree_decoder import TreeDecoder

class Parser:
    def __init__(self, original_model_id, peft_model_id,
                 is_instruct, representation_type, seed, model_library, max_tokens):
       self.llm = LLMInferencer(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens)
       self.tree_decoder = TreeDecoder(representation_type)

    def parse(self, sent):
        answer_output, full_output = self.llm.get_llm_output(sent)
        res = self.tree_decoder.decode_tree(answer_output)
        return answer_output, full_output, res
        
    def clear(self):
        del self.llm
        del self.tree_decoder

def inference_dataset(parser, filepath, result_filepath, index_set):
    with open(filepath, 'r') as f:
        data = json.load(f)
    res = []
    ts = time.time()
    for d_i, d in enumerate(data):
      if (index_set is None) or (d['index'] in index_set):
        new_d = { 'index': d['index'], 'input': d['input'], 'gold_output': d['output']}
        llm_output, full_output, pred_tree = parser.parse(d['input'])
        new_d['pred_output'] = llm_output
        new_d['full_pred_output'] = full_output
        new_d['pred_tree'] = pred_tree
        new_d['gold_tree'] = parser.tree_decoder.decode_tree(d['output'])
        res.append(new_d)
        print(f"{d_i}/{len(data)}. {time.time() - ts}")
    print(time.time() - ts)
    with open(result_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(res, json_file, ensure_ascii=False, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--config_name", nargs='?', default='/src/src/configs/config.yaml')
parser_args = parser.parse_args()
config_name = parser_args.config_name

with open(config_name, 'r') as file:
    configs = yaml.safe_load(file)
print(configs, "-" * 20, sep="\n")

output_dir = configs['output_dir']
seed = configs.get('seed', 42)

dataset_config = configs['dataset']
dataset_path = dataset_config['path']
dataset_name = dataset_config['name']
dataset_repr = dataset_config['representation_type']
index_set = dataset_config.get('index_set', None)
if index_set is not None:
    index_set = set(index_set)

for model_config in configs['models']:
    print(model_config)
    original_model_id = model_config['original_model_id']
    peft_model_id = model_config['peft_model_id']
    is_instruct = model_config['is_instruct']
    model_name = model_config['name']
    max_tokens = model_config.get('max_tokens', 512)
    model_library = model_config.get('model_library', 'transformers')
    result_path = f"{output_dir}/{model_name}_{dataset_name}.json"
    parser = Parser(original_model_id, peft_model_id, is_instruct, dataset_repr, seed, model_library, max_tokens)
    inference_dataset(parser, dataset_path, result_path, index_set)
    parser.clear()
    del parser
    for _ in range(3):
        gc.collect() # Сборка мусора для удаления
    torch.cuda.empty_cache()

