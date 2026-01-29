import argparse
import gc
import json
import os
import time
import yaml

import torch

from inference_functions.tree_decoder import TreeDecoder

class Parser:
    def __init__(self, original_model_id, peft_model_id,
                 is_instruct, representation_type, seed, model_library, max_tokens):
       if model_library == "guidance":
           from inference_functions.inferencer_guidance import LLMInferencerGuidance
           self.llm = LLMInferencerGuidance(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens)
       elif model_library == "vllm_xgrammar":
           from inference_functions.inferencer_vllm_xgrammar import LLMInferencerVllmXgrammar
           self.llm = LLMInferencerVllmXgrammar(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens)
       else:
           from inference_functions.inferencer import LLMInferencer
           self.llm = LLMInferencer(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens)
       self.tree_decoder = TreeDecoder(representation_type)

    def parse(self, input_text, input_tokens=None):
        ts = time.time()
        try:
            answer_output, full_output, token_amount = self.llm.get_llm_output(input_text, input_tokens)
        except Exception as e:
            print(f"Ошибка: {e}")
            answer_output, full_output, token_amount = None, None, None
        llm_time = time.time() - ts
        res = self.tree_decoder.decode_tree(answer_output)
        return answer_output, full_output, res, llm_time, token_amount
        
    def clear(self):
        del self.llm
        del self.tree_decoder

def inference_dataset(parser, filepath, result_filepath, index_set):
    with open(filepath, 'r') as f:
        data = json.load(f)
    res = []
    ts = time.time()
    last_saved_i = 0
    for d_i, d in enumerate(data):
        if (index_set is None) or (d['index'] in index_set):
            new_d = { 'index': d['index'], 'input': d['input'], 'gold_output': d['output']}
            new_d['gold_tree'] = parser.tree_decoder.decode_tree(d['output'])
            print(new_d['gold_tree'])
            input_tokens = [t['form'] for t in new_d['gold_tree']
                if '.' not in t['id']]
            llm_output, full_output, pred_tree, llm_time, token_amount = parser.parse(d['input'], input_tokens)
            new_d['pred_output'] = llm_output
            new_d['full_pred_output'] = full_output
            new_d['pred_tree'] = pred_tree
            new_d['llm_time'] = llm_time
            new_d['input_tokens'], new_d['output_tokens'] = token_amount
            res.append(new_d)
            print(f"{d_i}/{len(data)}. {time.time() - ts}")
        if len(res) - last_saved_i >= 10:
            with open(result_filepath, 'w', encoding='utf-8') as json_file:
                for s_i in range(last_saved_i + 1, len(res)):
                    json_file.write(json.dumps(res[s_i],
                        ensure_ascii=False) + '\n')
                json_file.flush()
            last_saved_i = len(res) - 1
    print(time.time() - ts)

def create_adapter_name(adapter_path):
    fragments = adapter_path.split("/")
    fragments = [fr for fr in fragments
                 if not fr.isdigit() and "checkpoint" not in fr]
    return fragments[-1]

def inference_main():
    os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "300"
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", nargs='?',
                        default='/src/src/configs/config.yaml')
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
        if "peft_model_id" in model_config:
            peft_adapters = [(model_config['peft_model_id'], model_config['name'])]
        else:
            config_adapters = model_config['peft_group']
            print(f"config_adapters:{config_adapters}")
            peft_adapters = [(a, create_adapter_name(a)) for a in config_adapters]
        for peft_model_id, adapter_name in peft_adapters:
            print(f"\npeft_model_id: {peft_model_id}\nadapter_name: {adapter_name}")
            is_instruct = model_config['is_instruct']
            max_tokens = model_config.get('max_tokens', 512)
            model_library = model_config.get('model_library', 'transformers')
            result_path = f"{output_dir}/{adapter_name}_{dataset_name}.jsonl"
            parser = Parser(original_model_id, peft_model_id, is_instruct,
                            dataset_repr, seed, model_library, max_tokens)
            inference_dataset(parser, dataset_path, result_path, index_set)
            parser.clear()
            del parser
            for _ in range(3):
                gc.collect() # Сборка мусора для удаления
            torch.cuda.empty_cache()

if __name__ == "__main__":
    inference_main()
