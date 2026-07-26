import transformers
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, GenerationConfig, set_seed
from peft import PeftModel


class TransformersModel:
    def __init__(self, original_model_id, peft_model_id, seed, max_tokens):
        set_seed(seed)
        self.max_tokens = max_tokens
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_from = AutoModelForCausalLM.from_pretrained(
            original_model_id,
            #load_in_4bit=True,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map={"": 0},
        )

        self.model = PeftModel.from_pretrained(
            model_from,
            peft_model_id
        ).to(model_from.device)
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.generation_config = GenerationConfig(
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

    def create_output(self, original_input_ids, input_tokens):
        input_ids = torch.tensor([original_input_ids]).to(self.model.device)
        with torch.no_grad():
            gen_outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                max_new_tokens=self.max_tokens,
                use_cache=True,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                early_stopping=True,  # Останавливает генерацию при первом eos_token
            )

        full_output_ids = gen_outputs[0]

        result_ids = full_output_ids[len(original_input_ids):]
        extra_info = None
        return full_output_ids, result_ids, extra_info
