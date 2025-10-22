from src.inference_functions.inferencer import LLMInferencer
from src.inference_functions.tree_decoder import TreeDecoder

class Parser:
    def __init__(self, original_model_id, peft_model_id,
                 is_instruct, representation_type):
       self.llm = LLMInferencer(original_model_id, peft_model_id, is_instruct)
       self.tree_decoder = TreeDecoder(representation_type)

    def parse(self, sent):
        answer_output, full_output = self.llm.get_llm_output(sent)
        res = self.tree_decoder.decode_tree(answer_output)
        return answer_output, full_output, res
        
import json

def inference_dataset(parser, filepath, result_filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    res = []
    for d in data:
        new_d = { 'index': d['index'], 'input': d['input'], 'gold_output': d['output']}
        llm_output, full_output, pred_tree = parser.parse(d['input'])
        new_d['pred_output'] = llm_output
        new_d['full_pred_output'] = full_output
        new_d['pred_tree'] = pred_tree
        new_d['gold_tree'] = parser.tree_decoder.decode_tree(d['output'])
        res.append(new_d)
    with open(result_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(res, json_file, ensure_ascii=False, indent=4)

