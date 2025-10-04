import argparse
import transformers
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import random
import os
import time
import yaml

random.seed(23)

from constants import LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, CUTOFF_LEN, WARMUP_RATIO
from deppllama_utils import *
from creating_data import creating_data
from parameters import Parameters
 
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
 
#import fire
import torch

#============================================
#               PARAMETERS
#============================================

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", nargs='?', default='/src/src/configs/config.yaml')
parser_args = parser.parse_args()
config_name = parser_args.config_name

with open(config_name, 'r') as file:
    configs = yaml.safe_load(file)

print(configs)

parameters = Parameters(configs)

os.makedirs(parameters.output_dir_path)
with open(parameters.output_dir_path + config_name.split('/')[-1], 'w') as file:
    yaml.dump(configs, file, default_flow_style=False)

print("LEARNING_RATE:\t" + str(parameters.learning_rate))



#============================================
#               FUNCTIONS
#============================================


 
# Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
def tokenize_base(prompt, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result

# Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
def tokenize(prompt, cutoff_len, add_eos_token=True):
    return tokenize_base(prompt, cutoff_len, add_eos_token)
 
# Notice: result["labels"] is rewritten so that only the output is considered
def generate_and_tokenize_prompt(data_point, add_eos_token=True):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, CUTOFF_LEN)

    user_prompt = generate_prompt_str(
        data_point["input"]
    )
    tokenized_user_prompt = tokenize(
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

    


def remove_example_by_length(lst, target_length):
    result = []
    for item in lst:
        if len(item["input_ids"])<target_length:
            result.append(item)
    return result

#============================================
#                   MAIN
#============================================

json_train, json_dev = creating_data(parameters)

#-------------------
#    LOAD MODEL
#-------------------
tokenizer = AutoTokenizer.from_pretrained(parameters.model_name, trust_remote_code=True)

# PREPARE DATA
train_data = ( json_train["train"].shuffle().map(generate_and_tokenize_prompt) )
val_data = ( json_dev["train"].shuffle().map(generate_and_tokenize_prompt) )


original_train_length = len(train_data)

train_data = remove_example_by_length(train_data, CUTOFF_LEN)

if(len(train_data)!=original_train_length):
    print("WARNING:")
    print("original_train_length: " + str(original_train_length))
    print("len(train_data): " + str(len(train_data)))

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if not parameters.disable_qlora:
    model = AutoModelForCausalLM.from_pretrained(
        parameters.model_name,
        #load_in_4bit=True,
        quantization_config=quant_config,
        #torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        parameters.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.half()

if "falcon" in parameters.model_name:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

print("padding_side\t" + str(tokenizer.padding_side))


# PREPARE MODEL
model = prepare_model_for_kbit_training(model)

if "falcon" in parameters.model_name:
    config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )
else:
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

model = get_peft_model(model, config)
model.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=parameters.micro_batch_size,
    gradient_accumulation_steps=parameters.gradient_accumulation_steps,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=parameters.epochs,
    learning_rate=parameters.learning_rate,
    fp16=True,
    logging_strategy = "steps",
    logging_steps=1,
    optim="paged_adamw_32bit",
    eval_strategy="epoch",
    save_strategy="epoch",
    output_dir=parameters.output_dir_path,
    save_total_limit=0,
    group_by_length=parameters.group_by_length,
    load_best_model_at_end=True,
    label_names=["labels"]
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

if "falcon" in parameters.model_name:
    model.config.pad_token_id = model.config.eos_token_id
else:
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

if torch.__version__ >= "2":
    print("YES! I AM 2.0 :-)")
    model = torch.compile(model)

#-------------------
#    LOAD MODEL
#-------------------
ts = time.time()
trainer.train()
print(f"Training time:{time.time() - ts}")

tokenizer.save_pretrained(parameters.output_dir_path)
model.save_pretrained(parameters.output_dir_path)
