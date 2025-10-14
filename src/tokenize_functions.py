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
    def tokenize(self, prompt):
        result = self.tokenizer(prompt, return_tensors=None)
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
        return result

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
    def __init__(self, parameters):
        self.tokenizer = AutoTokenizer.from_pretrained(
            parameters.model_config.model_name, trust_remote_code=True)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.tokenizer.padding_side = "left"
        print("padding_side\t" + str(self.tokenizer.padding_side))

    # Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
    def tokenize(self, messages, add_generation_prompt):
        result = self.tokenizer.apply_chat_template(messages, tokenize=True,
          add_generation_prompt=add_generation_prompt, return_tensors=None, enable_thinking=False,
          return_dict=True)
        #result = self.tokenizer(prompt, return_tensors=None)
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
        return result
      
    def generate_train_messages(self, data_point):
        return [
                  { "role": "user", "content": data_point['input'] },
                  { "role": "assistant", "content": data_point['output'] },
               ]

    def generate_pred_messages(self, data_point):
        return [
                  { "role": "user", "content": data_point['input'] },
               ]

    # Notice: result["labels"] is rewritten so that only the output is considered
    def generate_and_tokenize_prompt(self, data_point):
        train_messages = self.generate_train_messages(data_point)
        tokenized_train_messages = self.tokenize(train_messages, False)

        pred_messages = self.generate_pred_messages(data_point)
        tokenized_pred_messages = self.tokenize(pred_messages, True)
        tokenized_pred_messages_len = len(tokenized_pred_messages["input_ids"]) - 1 # Minus eos-token

        tokenized_train_messages["labels"] = [-100] * tokenized_pred_messages_len + \
            tokenized_train_messages["labels"][tokenized_pred_messages_len:]  # could be sped up, probably
        return tokenized_train_messages

