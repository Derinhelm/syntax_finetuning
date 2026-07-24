import importlib

import transformers
import torch

from tokenize_functions import BaseTokenizer, InstructTokenizer
from inference_functions.vllm_creating_logit_processor import create_logit_processor

class LLMInferencer:
    def __init__(self, original_model_id, peft_model_id, is_instruct,
                 seed, model_library, max_tokens, logit_parameters, prompt_name):
        self.model_library = model_library    
        if is_instruct:
            self.tokenizer = InstructTokenizer(original_model_id)
        else:
            self.tokenizer = BaseTokenizer(original_model_id)

        if prompt_name is not None:
            module_name = "inference_functions.prompt_creating.prompt_functions"
            module = importlib.import_module(module_name)
            if hasattr(module, prompt_name):
                self.prompt_fun = getattr(module, prompt_name)
            else:
                print(f"No prompt function {prompt_name}")
                self.prompt_fun = lambda input_text_param, _: input_text_param
        else:
            self.prompt_fun = lambda input_text_param, _: input_text_param

        if model_library == "transformers":
            from inference_functions.transformers_model import TransformersModel
            self.model = TransformersModel(original_model_id, peft_model_id, seed, max_tokens)
        elif model_library == "vllm":
            from inference_functions.vllm_model import VllmModel
            if logit_parameters != {}:
               assert peft_model_id is not None
               logit_processor = create_logit_processor(logit_parameters, self.tokenizer.tokenizer)
            else:
               logit_processor = None
            self.model = VllmModel(original_model_id, peft_model_id, seed, max_tokens, logit_processor)
        else:
            print(f"Error model_library:{model_library}")

    def get_llm_output(self, input_text_param, input_tokens):
        input_text = self.prompt_fun(input_text_param, input_tokens)
        print(f"Creating prompt.\n\ninput_text_param:{input_text_param}",
              f"input_tokens:{input_tokens}\n",
              f"After prompt:{input_text}", sep="\n")
        inputs = self.tokenizer.encode_input(input_text)
        input_ids = inputs['input_ids']
        if input_ids[-1] == self.tokenizer.tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
        output, extra_info = self.model.create_output(input_ids, input_tokens)
        full_output = self.tokenizer.tokenizer.decode(output)
        result_ids = output[len(input_ids):]
        if result_ids[-1] == self.tokenizer.tokenizer.eos_token_id:
            result_ids = result_ids[:-1]
        result = self.tokenizer.decode(result_ids).rstrip().lstrip()
        token_amount = (len(input_ids), len(result_ids))
        return result, full_output, token_amount, extra_info
