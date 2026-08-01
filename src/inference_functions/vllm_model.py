import copy
import signal

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest
import torch

def timeout_handler(signum, frame): # TODO: - убрать дублирование
    raise TimeoutError("Время выполнения истекло!")


class VllmModel:
    def __init__(self, original_model_id, peft_model_id, sampling_params,
                seed, max_tokens,
                logit_processor, stop_prefix):
        # peft_model_id should be a path to directory with lora adapter
        enable_lora = peft_model_id is not None
        print(f"enable_lora: {enable_lora}")
        self.llm = LLM(
            seed=seed,
            model=original_model_id,
            max_seq_len_to_capture=4096, # TODO
            max_model_len=4096,
            dtype=torch.float16,
            enable_lora=enable_lora,
            skip_tokenizer_init=True,
            enforce_eager=True, # Для воспроизводимости
        )
        self.seed = seed
        self.max_tokens = max_tokens # TODO: В sampling_params
        self.sampling_params_info = sampling_params
        print(f"sampling_params: {sampling_params}")
        self.peft_model_id = peft_model_id # TODO
        self.logit_processor = logit_processor
        self.stop_prefix = stop_prefix
        print(f"stop_prefix: {stop_prefix}")
        signal.signal(signal.SIGALRM, timeout_handler)

    def create_output(self, input_ids, input_tokens):
        if self.logit_processor is not None:
            self.logit_processor.create_new_context(max_op_bracket=len(input_tokens))
        tokens_prompt = TokensPrompt(prompt_token_ids=input_ids)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_tokens,
            stop_token_ids=[2],
            seed=self.seed,
            logits_processors = [self.logit_processor] if self.logit_processor is not None else None
        )
        for k, v in self.sampling_params_info.items():
            sampling_params.__setattr__(k, v)

        signal.alarm(120)
        self.llm.llm_engine.scheduler[0].waiting.clear()  # Очистить ожидающие
        self.llm.llm_engine.scheduler[0].running.clear()  # Очистить выполняющиеся
        self.llm.llm_engine.scheduler[0].swapped.clear()  # Очистить swap
        if self.peft_model_id is not None:
            lora_request = LoRARequest("lora_adapter", 1, self.peft_model_id)
        else:
            lora_request = None
        if self.stop_prefix is not None:
            prefix_sampling_params = copy.deepcopy(sampling_params)
            prefix_sampling_params.stop=[self.stop_prefix]
            prefix_sampling_params.include_stop_str_in_output=True
            prefix_outputs = self.llm.generate(
                [tokens_prompt],
                sampling_params=prefix_sampling_params,
                lora_request=lora_request,
                )
            prefix_ids = list(prefix_outputs[0].outputs[0].token_ids)
            tokens_prompt_with_prefix = TokensPrompt(
                prompt_token_ids=input_ids + prefix_ids)
        else:
            prefix_ids = [] 
            tokens_prompt_with_prefix = tokens_prompt

        outputs = self.llm.generate(
           [tokens_prompt_with_prefix],
           sampling_params=sampling_params,
           lora_request=lora_request,
        )
        signal.alarm(0)
        extra_info = list(outputs[0].outputs[0].token_ids)
        full_output_ids = outputs[0].prompt_token_ids + prefix_ids + list(outputs[0].outputs[0].token_ids)
        result_ids = list(outputs[0].outputs[0].token_ids)
        return full_output_ids, result_ids, extra_info
