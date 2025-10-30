import transformers
import torch

from tokenize_functions import BaseTokenizer, InstructTokenizer

class LLMInferencer:
    def __init__(self, original_model_id, peft_model_id, is_instruct, seed, model_library="transformers"):
        self.model_library = model_library
        if model_library == "transformers":
            from inference_functions.transformers_model import TransformersModel
            self.model = TransformersModel(original_model_id, peft_model_id, seed)
        elif model_library == "vllm":
            from inference_functions.vllm_model import VllmModel
            self.model = VllmModel(original_model_id, peft_model_id, seed)
        else:
            print(f"Error model_library:{model_library}")
        
        if is_instruct:
            self.tokenizer = InstructTokenizer(original_model_id)
        else:
            self.tokenizer = BaseTokenizer(original_model_id)


    def get_llm_output(self, input_text):
        inputs = self.tokenizer.encode_input(input_text)
        input_ids = inputs['input_ids']
        if input_ids[-1] == self.tokenizer.tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
        output = self.model.create_output(input_ids)
        full_output = self.tokenizer.tokenizer.decode(output)
        result_ids = output[len(input_ids):]
        if result_ids[-1] == self.tokenizer.tokenizer.eos_token_id:
            result_ids = result_ids[:-1]
        result = self.tokenizer.decode(result_ids).rstrip().lstrip()
        return result, full_output
        
            

