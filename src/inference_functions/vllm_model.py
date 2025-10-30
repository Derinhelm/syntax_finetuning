from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
import torch

class VllmModel:
    def __init__(self, original_model_id, lora_path, seed):
        # lora_path should be a path to directory with lora adapter
        self.llm = LLM(
            seed=seed,
            model=original_model_id,
            max_seq_len_to_capture=4096, # TODO
            max_model_len=4096,
            dtype=torch.float16,
            enable_lora=True
        )
        self.seed = seed
        self.lora_path = lora_path # TODO

    def create_output(self, input_ids):
        tokens_prompt = TokensPrompt(prompt_token_ids=input_ids)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=512,
            stop_token_ids=[2],
            seed=self.seed,
            n=1,
            best_of=4,
            use_beam_search=True,
            early_stopping=True,
        )
        outputs = llm.generate(
           [tokens_prompt],
           sampling_params=sampling_params,
           lora_request=LoRARequest("lora_adapter", 1, self.lora_path)
        )
        return outputs[0].outputs[0].text.strip(), None
