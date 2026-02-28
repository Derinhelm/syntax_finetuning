from copy import deepcopy
import signal

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from inference_functions.prompt_creating.prompt_functions import prompt2

def timeout_handler(signum, frame):
    raise TimeoutError("Время выполнения истекло!")

class LLMInferencerVllmPartRegex:
    def __init__(self, original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens):
        self.model_library = "vllm_xgrammar"
        assert peft_model_id is None
        self.model = LLM(model=original_model_id, dtype=torch.float16,
            max_model_len=max_tokens, seed=seed,
            enable_prefix_caching=True,)
        self.max_tokens = max_tokens
        self.seed = seed
        signal.signal(signal.SIGALRM, timeout_handler)
        self.tokenizer = self.model.get_tokenizer()
        self.relations = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc',
             'ccomp', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det',
             'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'iobj', 'list',
             'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan',
             'parataxis', 'punct', 'root', 'vocative', 'xcomp']

    def get_llm_output(self, input_text, input_tokens=None):
        tokens_text = " ".join(input_tokens)
        conll_prompt = prompt2(tokens_text, input_tokens)

        messages = [
            {"role": "user", "content": conll_prompt},
            {"role": "assistant", "content": ""}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
        )
        source_tokens = self.tokenizer.encode(prompt)
        prompt_tokens = deepcopy(source_tokens)

        # Guided decoding by Grammar
        parent_ids = [str(i) for i in range(0, len(input_tokens) + 1)]
        parent_id_grammar = '?start: "' + '" | "'.join(parent_ids) + '"\n'
        relation_grammar = '?start: "' + '" | "'.join(self.relations) + '"\n'

        guided_decoding_params_grammar_parent_ids = GuidedDecodingParams(
            grammar=parent_id_grammar)
        sampling_params_grammar_parent_ids = SamplingParams(
            guided_decoding=guided_decoding_params_grammar_parent_ids,
            best_of=5, n=1,
            seed=self.seed,
            stop_token_ids=[self.tokenizer.eos_token_id],)

        guided_decoding_params_grammar_relations = GuidedDecodingParams(
            grammar=relation_grammar)
        sampling_params_grammar_relations = SamplingParams(
            guided_decoding=guided_decoding_params_grammar_relations,
            best_of=5, n=1,
            seed=self.seed,
            stop_token_ids=[self.tokenizer.eos_token_id],)

        new_line_tokens = self.tokenizer.encode("\n")
        space_tokens = self.tokenizer.encode(" ")

        for t_i, t in enumerate(input_tokens):
            part_line = self.tokenizer.encode(f"{t_i + 1} {t}") # TODO: Реализация для prompt2
            prompt_tokens += part_line
            tokens_prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
            signal.alarm(120)
            llm_output = self.model.generate([tokens_prompt],
                sampling_params=sampling_params_grammar_parent_ids)
            signal.alarm(0)
            parent_id_tokens = [t for t in llm_output[0].outputs[0].token_ids
                if t != self.tokenizer.eos_token_id]
            prompt_tokens += parent_id_tokens + space_tokens
            tokens_prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
            signal.alarm(120)
            llm_output = self.model.generate([tokens_prompt],
                sampling_params=sampling_params_grammar_relations)
            signal.alarm(0)
            relation_tokens = [t for t in llm_output[0].outputs[0].token_ids
                if t != self.tokenizer.eos_token_id]
            prompt_tokens += relation_tokens + new_line_tokens

        source_len = len(source_tokens)
        pred_tokens = prompt_tokens[source_len:]
        result = self.tokenizer.decode(pred_tokens)
        full_output = self.tokenizer.decode(prompt_tokens)
        return result, full_output, (len(source_tokens), len(pred_tokens))
