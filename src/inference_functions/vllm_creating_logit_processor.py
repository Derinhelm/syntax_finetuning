class OriginalLogitsProcessor:
    def __init__(self, tokenizer, logit_params):
        self.logit_params = logit_params
        self.max_op_bracket = None

    def create_new_context(self, input_tokens): 
        self.max_op_bracket = len(input_tokens)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, token_ids, logits):
        return logits


def create_logit_processor(logit_params, tokenizer):
    logit_name = logit_params.get("name", "")
    if logit_name == "structural_logits": # TODO: to delete
        from inference_functions.vllm_creating_logit_structural_processor import StructuralLogitsProcessor
        logit_processor = StructuralLogitsProcessor(tokenizer, logit_params)
    elif logit_name == "main_logits": # Без текста тоже здесь
        from inference_functions.vllm_creating_logit_text_processor import TextBracketLogitsProcessor
        logit_processor = TextBracketLogitsProcessor(tokenizer, logit_params)
    elif logit_name == "original_logits":
        logit_processor = OriginalLogitsProcessor(tokenizer, logit_params)
    else:
        print(f"Wrong logit processor: {logit_name}, use OriginalLogitsProcessor")
        logit_processor = OriginalLogitsProcessor(tokenizer, logit_params)

    return logit_processor
