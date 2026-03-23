class BracketLogitsProcessor:
    def __init__(self, op_code, end_code, eos_id, bracket_codes):
        self.op_code = op_code
        self.end_code = end_code
        self.max_op_bracket = None
        self.eos_id = eos_id
        self.tokenizer = None
        self.bracket_codes = bracket_codes

    def set_max_op_bracket(self, max_op_bracket):
        self.max_op_bracket = max_op_bracket * 2

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def restrict_end(self, logits):
        for token_text, token_id in self.bracket_codes:
            if "]" in token_text:
              logits[token_id] = float('-inf')
              #print(token_id, token_text)
        return logits

    def restrict_open(self, logits):
        for token_text, token_id in self.bracket_codes:
            if "[" in token_text:
              logits[token_id] = float('-inf')
              #print(token_id, token_text)
        return logits

    def restrict_all(self, logits):
        for token_text, token_id in self.bracket_codes:
            logits[token_id] = float('-inf')
            #print(token_id, token_text)
        return logits

    def __call__(self, token_ids, logits):
        #print(token_ids)
        logits = logits.clone()
        if len(token_ids) == 0: # First subtoken has to be "[" # TODO: or startswith "["
            logits[:self.op_code] = float('-inf')
            logits[self.op_code + 1:] = float('-inf')
        else:
            generated_text = self.tokenizer.decode(token_ids)
            #print(generated_text)
            op_amount = generated_text.count("[")
            end_amount = generated_text.count("]")
            #print(op_amount, end_amount)
            if op_amount == self.max_op_bracket:
                if end_amount == self.max_op_bracket: # Finish generating
                    logits[:self.eos_id] = float('-inf')
                    logits[self.eos_id + 1:] = float('-inf')
                    #print("Finish restriction")
                else:
                    logits = self.restrict_open(logits)
                    #print("Restriction for [")
                    if generated_text[-1] == "]": # Generate some last "]"
                        logits[:self.end_code] = float('-inf')
                        logits[self.end_code + 1:] = float('-inf')
                        #print("Allowing only ]")
            else:
                if generated_text[-1] == "[": # After "[" any bracket subtoken is restricted
                    logits = self.restrict_all(logits)
                    #print("Restriction for bracket after open bracket")
                elif op_amount - end_amount <= 1:
                    logits = self.restrict_end(logits)
                    #print("Restriction for ] because of small amount of [")
        #print()
        return logits



def create_logit_processor(logit_params, tokenizer):
    vocab = tokenizer.get_vocab()
    bracket_tokens = [(k, v) for k, v in vocab.items()
                      if set(k) <= {"[", "]"} and "[]" not in k]
    #print(bracket_tokens)

    op_code = tokenizer.encode("[")
    assert len(op_code) == 1
    op_code = op_code[0]
    end_code = tokenizer.encode("]")
    assert len(end_code) == 1
    end_code = end_code[0]

    eos_code = 2 # TODO: Сделать константу, используется в других местах

    #print(op_code, end_code, eos_code)

    logit_processor = BracketLogitsProcessor(op_code, end_code, eos_code, bracket_tokens)
    logit_processor.set_tokenizer(tokenizer)
    logit_processor.set_tokenizer(tokenizer)
    return logit_processor
