from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
import torch

class VllmModel:
    def __init__(self, original_model_id, peft_model_id, seed, max_tokens, logit_processor):
        # peft_model_id should be a path to directory with lora adapter
        self.llm = LLM(
            seed=seed,
            model=original_model_id,
            max_seq_len_to_capture=4096, # TODO
            max_model_len=4096,
            dtype=torch.float16,
            enable_lora=True,
            skip_tokenizer_init=True,
        )
        self.seed = seed
        self.max_tokens = max_tokens
        self.peft_model_id = peft_model_id # TODO
        self.logit_processor = logit_processor

    def create_output(self, input_ids, input_tokens):
        if self.logit_processor is not None:
            self.logit_processor.set_max_op_bracket(len(input_tokens))
        tokens_prompt = TokensPrompt(prompt_token_ids=input_ids)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_tokens,
            stop_token_ids=[2],
            seed=self.seed,
            logits_processors = [self.logit_processor] if self.logit_processor is not None else None
        )
        outputs = self.llm.generate(
           [tokens_prompt],
           sampling_params=sampling_params,
           lora_request=LoRARequest("lora_adapter", 1, self.peft_model_id),
        )
        return outputs[0].prompt_token_ids + list(outputs[0].outputs[0].token_ids)
