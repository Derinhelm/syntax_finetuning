import transformers
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from peft import PeftModel

from src.tokenize_functions import BaseTokenizer, InstructTokenizer

class LLMInferencer:
    def __init__(self, original_model_id, peft_model_id, is_instruct, seed):
        set_seed(seed)
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
        )
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.generation_config = GenerationConfig(
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )
        if is_instruct:
            self.tokenizer = InstructTokenizer(original_model_id)
        else:
            self.tokenizer = BaseTokenizer(original_model_id)


    def get_llm_output(self, input_text):
        inputs = self.tokenizer.encode_input(input_text)
        input_ids = inputs['input_ids']
        if input_ids[-1] == self.model.config.eos_token_id:
            input_ids = input_ids[:-1]
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        with torch.no_grad():
            gen_outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                max_new_tokens=1024,
                use_cache=True,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                early_stopping=True,  # Останавливает генерацию при первом eos_token
            )

        output_ids = gen_outputs[0]
        full_output = self.tokenizer.tokenizer.decode(output_ids)

        result_ids = output_ids[len(input_ids[0]):]

        if result_ids[-1] == self.model.config.eos_token_id:
            result_ids = result_ids[:-1]
        result = self.tokenizer.decode(result_ids).rstrip().lstrip()
        return result, full_output

