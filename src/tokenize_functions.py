from transformers import AutoTokenizer
from deppllama_utils import *

class BaseTokenizer:
    def __init__(self, parameters):
        self.tokenizer = AutoTokenizer.from_pretrained(
            parameters.model_config.model_name, trust_remote_code=True)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.tokenizer.padding_side = "left"
        print("padding_side\t" + str(self.tokenizer.padding_side))

    # Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
    def tokenize_base(self, prompt):
        result = self.tokenizer(prompt, return_tensors=None)
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
        return result

    # Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
    def tokenize(self, prompt):
        return self.tokenize_base(prompt)

    # Notice: result["labels"] is rewritten so that only the output is considered
    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)

        user_prompt = generate_prompt_str(data_point["input"])
        tokenized_user_prompt = self.tokenize(user_prompt)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) - 1 # Minus eos-token

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + \
            tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
        return tokenized_full_prompt

class InstructTokenizer:
    pass
