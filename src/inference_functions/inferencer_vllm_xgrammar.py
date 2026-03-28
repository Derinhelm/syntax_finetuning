import signal
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from inference_functions.prompt_creating.prompt_functions import prompt2

def timeout_handler(signum, frame):
    raise TimeoutError("Время выполнения истекло!")

class LLMInferencerVllmXgrammar:
    def __init__(self, original_model_id, peft_model_id, is_instruct, seed, model_library, max_tokens):
        self.model_library = "vllm_xgrammar"
        self.model = LLM(model=original_model_id, dtype=torch.float16,
            max_model_len=max_tokens, seed=seed)
        self.max_tokens = max_tokens
        self.seed = seed
        signal.signal(signal.SIGALRM, timeout_handler)

    def get_llm_output(self, input_text, input_tokens=None):
        print(f"input_tokens: {input_tokens}")

        relations = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc',
             'ccomp', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det',
             'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'iobj', 'list',
             'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan',
             'parataxis', 'punct', 'root', 'vocative', 'xcomp']
        ids = [str(i) for i in range(len(input_tokens) + 1)]
        conll_grammar = ""
        conll_grammar += "root ::= " +  ' "\\n" '.join([f"line{id}" for id in ids]) + "\n"
        for t_i, t in enumerate(input_tokens):
            t_change = t.replace('\"', '\\"')
            r_line = f'line{t_i + 1} ::= "{t_i + 1} {t_change} " id " " rel' + "\n"
            conll_grammar += r_line
        conll_grammar += 'id ::= "' + '" | "'.join([0] + ids) + '"\n'
        conll_grammar += 'rel ::= "' + '" | "'.join(relations) + '"\n'
        #print(conll_grammar)

        # Guided decoding by Grammar

        guided_decoding_params_grammar = GuidedDecodingParams(
            grammar=conll_grammar)
        sampling_params_grammar = SamplingParams(
            guided_decoding=guided_decoding_params_grammar,
            max_tokens=self.max_tokens,
            temperature=0,
            seed=self.seed)

        # TODO: В исходном промпте перечисляем не текст, а токены через ' '
        conll_prompt = prompt2(input_text, input_tokens)

        signal.alarm(120)
        outputs = self.model.generate(prompts=conll_prompt, sampling_params=sampling_params_grammar)
        signal.alarm(0)
        #print(f"res: {outputs[0].outputs[0].text}")
        result = outputs[0].outputs[0].text
        full_output = outputs[0].prompt + result
        extra_info = None
        return result, full_output, (len(outputs[0].prompt_token_ids), len(outputs[0].outputs[0].token_ids)), extra_info

