import gc
import json
import os
import sys
import time
import traceback

import torch

import resource

def create_decoder(representation_type):
    if representation_type == "conll_short":
        from inference_functions.tree_decoder_conll_short import TreeDecoderConllShort
        return TreeDecoderConllShort(representation_type)
    else:
        from inference_functions.tree_decoder import TreeDecoder
        return TreeDecoder(representation_type)

class Parser:
    def __init__(self, original_model_id, peft_model_id,
                 is_instruct, representation_type, seed, model_library, max_tokens,
                 representation_type_result, logit_parameters):
        if model_library == "guidance":
           from inference_functions.inferencer_guidance import LLMInferencerGuidance
           self.llm = LLMInferencerGuidance(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens)
        elif model_library == "vllm_xgrammar":
           from inference_functions.inferencer_vllm_xgrammar import LLMInferencerVllmXgrammar
           self.llm = LLMInferencerVllmXgrammar(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens)
        elif model_library == "vllm_part_regex":
           from inference_functions.inferencer_vllm_part_regex import LLMInferencerVllmPartRegex
           self.llm = LLMInferencerVllmPartRegex(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens)
        else:
           from inference_functions.inferencer import LLMInferencer
           self.llm = LLMInferencer(original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens, logit_parameters)
        self.tree_decoder = create_decoder(representation_type)
        if representation_type_result is not None:
            self.tree_decoder_result = create_decoder(representation_type_result)
        else:
            # Связывание с существующим декодером
            self.tree_decoder_result = self.tree_decoder

    def parse(self, input_text, input_tokens=None):
        ts = time.time()
        try:
            answer_output, full_output, token_amount, extra_info = self.llm.get_llm_output(input_text, input_tokens)
        except Exception as e:
            print(f"Ошибка: {e}")
            print(traceback.print_exc())
            answer_output, full_output, token_amount, extra_info = None, None, (None, None), None
        llm_time = time.time() - ts
        res = self.tree_decoder_result.decode_tree(answer_output)
        return answer_output, full_output, res, llm_time, token_amount, extra_info
        
    def clear(self):
        del self.llm
        if self.tree_decoder_result != self.tree_decoder:
            del self.tree_decoder_result
        del self.tree_decoder

def inference_dataset(parser, filepath, result_filepath, index_predicate_param):
    last_ready_index = None
    try:
        with open(result_filepath, 'r') as f:
            lines = f.readlines()
            if lines:
                last_ready_index = json.loads(lines[-1])["index"]
                index_predicate = lambda ind2: ind2 > last_ready_index and index_predicate_param(ind2)
            else:
                index_predicate = index_predicate_param
            print(f"{result_filepath} exists, last_ready_index: {last_ready_index}")
    except FileNotFoundError:
        with open(result_filepath, 'x') as f:
            pass # Creating file, if not exist
        index_predicate = index_predicate_param
        print(f"Creating {result_filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    res = []
    ts = time.time()
    last_unsaved_i = 0
    for d_i, d in enumerate(data):
        if index_predicate(d['index']):
            new_d = { 'index': d['index'], 'input': d['input'], 'gold_output': d['output']}
            new_d['gold_tree'] = parser.tree_decoder.decode_tree(d['output'])
            print(new_d['gold_tree'])
            input_tokens = [t['form'] for t in new_d['gold_tree']
                if '.' not in t['id']]
            llm_output, full_output, pred_tree, llm_time, token_amount, extra_info = parser.parse(d['input'], input_tokens)
            new_d['pred_output'] = llm_output
            new_d['full_pred_output'] = full_output
            new_d['pred_tree'] = pred_tree
            new_d['llm_time'] = llm_time
            new_d['input_tokens'], new_d['output_tokens'] = token_amount
            new_d['extra_info'] = extra_info
            res.append(new_d)
            print(f"{d_i}/{len(data)}. {time.time() - ts}")
        if len(res) - last_unsaved_i >= 10:
            with open(result_filepath, 'a', encoding='utf-8') as json_file:
                for s_i in range(last_unsaved_i, len(res)):
                    json_file.write(json.dumps(res[s_i],
                        ensure_ascii=False) + '\n')
                json_file.flush()
            last_unsaved_i = len(res)
    with open(result_filepath, 'a', encoding='utf-8') as json_file:
        for s_i in range(last_unsaved_i, len(res)):
            json_file.write(json.dumps(res[s_i],
                ensure_ascii=False) + '\n')
            json_file.flush()
    print(time.time() - ts)

def create_adapter_name(adapter_path):
    fragments = adapter_path.split("/")
    fragments = [fr for fr in fragments
                 if not fr.isdigit() and "checkpoint" not in fr]
    return fragments[-1]

def start_inference_experiment(exp):
    index_predicate = exp['data_restriction_config'].create_index_predicate()

    model_config = exp['model_config']
    print(f"\npeft_model_id: {exp['peft_model_id']}\nadapter_name: {model_config['adapter_name']}")
    is_instruct = model_config['is_instruct']
    max_tokens = model_config.get('max_tokens', 512)
    model_library = model_config.get('model_library', 'transformers')
    logit_params = exp['cur_parameters'].logit_parameters
    logits_name = logit_params['name'] if logit_params != {} else 'logits_None'
    dataset_name = exp['dataset_config'].treebank
    dataset_path = exp['dataset_config'].test_file_path
    dataset_repr = exp['dataset_config'].treebank_repr
    seed = exp['cur_parameters'].seed

    result_path = f"{exp['output_dir']}/{model_config['adapter_name']}_{dataset_name}_{seed}_{logits_name}.jsonl"
    representation_type_result = model_config.get('representation_type_result')
    parser = Parser(model_config['original_model_id'],
        model_config['peft_model_id'], is_instruct,
        dataset_repr, seed, model_library, max_tokens,
        representation_type_result, logit_parameters=logit_params)
    inference_dataset(parser, dataset_path, result_path, index_predicate)
    parser.clear()
    del parser
    for _ in range(3):
        gc.collect() # Сборка мусора для удаления
    torch.cuda.empty_cache()
