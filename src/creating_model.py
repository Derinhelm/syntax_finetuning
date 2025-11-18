from transformers import BitsAndBytesConfig, AutoModelForCausalLM

from constants import LORA_TARGET_MODULES
from deppllama_utils import *
 
from peft import (
    LoraConfig,
    LoftQConfig,
    get_peft_model,
    replace_lora_weights_loftq,
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

    model = AutoModelForCausalLM.from_pretrained(
            parameters.model_config.model_name,
            #load_in_4bit=True,
            quantization_config=quant_config,
            #torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

    # PREPARE MODEL
    model = prepare_model_for_kbit_training(model)

    #loftq_config = None
    #if parameters.init_lora_weights == "loftq":
    #    loftq_config = LoftQConfig(loftq_bits=4)
    config = LoraConfig(
            r=parameters.lora_r,
            lora_alpha=parameters.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=parameters.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            #init_lora_weights=parameters.init_lora_weights,
            #loftq_config = loftq_config,
    )

    model = get_peft_model(model, config)
    replace_lora_weights_loftq(model)
    model.print_trainable_parameters()
    return model
