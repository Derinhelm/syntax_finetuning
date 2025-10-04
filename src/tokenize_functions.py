from transformers import AutoTokenizer
from constants import CUTOFF_LEN
from deppllama_utils import *

class Tokenizer:
    def __init__(self, parameters):
        self.tokenizer = AutoTokenizer.from_pretrained(
            parameters.model_name, trust_remote_code=True)

    # Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
    def tokenize_base(self, prompt, cutoff_len, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
    
        return result

    # Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
    def tokenize(self, prompt, cutoff_len, add_eos_token=True):
        return self.tokenize_base(prompt, cutoff_len, add_eos_token)

    # Notice: result["labels"] is rewritten so that only the output is considered
    def generate_and_tokenize_prompt(self, data_point, add_eos_token=True):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt, CUTOFF_LEN)

        user_prompt = generate_prompt_str(
            data_point["input"]
        )
        tokenized_user_prompt = self.tokenize(
            user_prompt, CUTOFF_LEN, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
        return tokenized_full_prompt

    def set_tokens(self, parameters):
        if "falcon" in parameters.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.pad_token_id = 0
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2
            self.tokenizer.padding_side = "left"
        print("padding_side\t" + str(self.tokenizer.padding_side))
