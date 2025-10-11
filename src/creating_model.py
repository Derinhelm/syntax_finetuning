from transformers import BitsAndBytesConfig, AutoModelForCausalLM

from constants import LORA_TARGET_MODULES
from deppllama_utils import *
 
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
 
#import fire
import torch

def creating_model(parameters):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if not parameters.disable_qlora:
        model = AutoModelForCausalLM.from_pretrained(
            parameters.model_config.model_name,
            #load_in_4bit=True,
            quantization_config=quant_config,
            #torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            parameters.model_config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.half()


    # PREPARE MODEL
    model = prepare_model_for_kbit_training(model)

    if "falcon" in parameters.model_config.model_name:
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
            r=parameters.lora_r,
            lora_alpha=parameters.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=parameters.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model