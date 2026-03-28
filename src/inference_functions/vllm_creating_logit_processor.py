from abc import ABC, abstractmethod
import string
from typing import List, Tuple


class Constraint(ABC):
    """Базовый класс для всех ограничений"""
    
    @abstractmethod
    def __call__(self, logits: torch.Tensor, processor, context) -> torch.Tensor:
        """
        Применяет ограничение к логитам
        
        Returns:
            измененные логиты
        """
        pass
       
    @abstractmethod
    def check(self, context):
        pass
        
# =============================================
# Force constraints

class ForceFirstTokenConstraint(Constraint):
    """Первый токен должен быть '['"""
    
    def __init__(self, op_code):
        self.op_code = op_code
        
    def check(self, context):
    # First subtoken has to be "["
    # TODO: or startswith "["
        return len(context.token_ids) == 0
    
    def __call__(self, logits, context):
        logits[:self.op_code] = float('-inf')
        logits[self.op_code + 1:] = float('-inf')
        return logits
        
class ForceRootPrefixConstraint(Constraint):
     def __init__(self, first_root, tokenizer):
         self.first_root = first_root
         print(f"first_root: {self.first_root}")
         self.root_prefix = "[root["
         self.tokenizer = tokenizer

     def check(self, context):
         return self.first_root and len(context.generated_text) <= 6
         # "[root[" - 6 symbols
         
     def __call__(self, logits, context): # TODO: можно ли убрать context ?
         print("ForceRootPrefixConstraint")
         for tok_id, _ in enumerate(logits):
         # TODO: for optimization
             potential_new_text = context.generated_text + self.tokenizer.decode(tok_id)
             min_pair_len = min(len(potential_new_text), len(self.root_prefix))
             if potential_new_text[:min_pair_len] != self.root_prefix[:min_pair_len]:
                 logits[tok_id] = float('-inf')
         return logits

class ForceEndConstraint(Constraint):
    """"""
    
    def __init__(self, end_code): # TODO: Разрешить "]]"
        self.end_code = end_code
        
    def check(self, context):
        return context.op_amount == context.max_op_bracket and context.end_amount != context.max_op_bracket and context.generated_text[-1] == "]"
        # Generate some last "]"

    def __call__(self, logits, context):
        logits[:self.end_code] = float('-inf')
        logits[self.end_code + 1:] = float('-inf')
        print("Allowing only ]")
        return logits

class ForceFinishConstraint(Constraint):
    """"""
    def __init__(self, eos_id):
        self.eos_id = eos_id
    
    def check(self, context):
        return context.op_amount == context.max_op_bracket and context.end_amount == context.max_op_bracket # Finish generating
    
    def __call__(self, logits, context):
        logits[:self.eos_id] = float('-inf')
        logits[self.eos_id + 1:] = float('-inf')
        print("Finish restriction")
        return logits
# =============================================
# Restricted constraints

class RestrictBracketAfterOpenConstraint(Constraint):
    def __init__(self, partial_bracket_codes):
        self.partial_bracket_codes = partial_bracket_codes

    def check(self, context):
        return context.generated_text[-1] == "["
        # After "[" any bracket subtoken is restricted
        
    def __call__(self, logits, context):
        for token_text, token_id in self.partial_bracket_codes:
            if token_text[0] in {"[", "]"}:
                logits[token_id] = float('-inf')
                #print(token_id, token_text)
        print("Restriction for bracket after open bracket")
        return logits
        
class RestrictBalanceBracketConstraint(Constraint):
    """"""
    
    def __init__(self, partial_bracket_codes):
        self.partial_bracket_codes = partial_bracket_codes
        
    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        bracket_diff = context.op_amount - context.end_amount
        for token_text, token_id in self.partial_bracket_codes:
          new_bracket_diff = bracket_diff + token_text.count("[") - token_text.count("]")
          full_open = context.op_amount == context.max_op_bracket
          # full_open - сгенерированы все возможные открытые скобки
          # new_bracket_diff может быть отрицательной, для "]]]]"
          if (not full_open and new_bracket_diff < 1) or \
              (full_open and new_bracket_diff < 0):
              logits[token_id] = float('-inf')
              #print(f"Restriction for {token_text} because of small amount of [")
              
        return logits

class RestrictOpenConstraint(Constraint):
    """"""
    
    def __init__(self, partial_bracket_codes):
        self.partial_bracket_codes = partial_bracket_codes
        
    def check(self, context):
        return context.op_amount == context.max_op_bracket and context.end_amount != context.max_op_bracket
    
    def __call__(self, logits, context):
        print("Restriction for [")
        for token_text, token_id in self.partial_bracket_codes:
            if "[" in token_text:
              logits[token_id] = float('-inf')
              #print(f"Restricted {token_id} ({token_text})")
        return logits

class RestrictErrorTokenConstraint(Constraint):
    """"""
    
    def __init__(self, partial_bracket_codes, tokenizer):
        self.error_indexes = []
        for token_text, token_id in partial_bracket_codes:
            error_token_flag = False
            decode_token_text = tokenizer.decode([token_id])
            for t_i, t in enumerate(decode_token_text):
               if t == "]":
                   if t_i != len(decode_token_text) - 1 and \
                     decode_token_text[t_i + 1] not in {"]", "["}:
                       error_token_flag = True
                       break
               elif t == "[":
                   if t_i != len(decode_token_text) - 1 and \
                     decode_token_text[t_i + 1] == "]": # "[]"
                       error_token_flag = True
                       break
                   if t_i != 0 and decode_token_text[t_i - 1] != "]":
                       error_token_flag = True
                       break
               elif not (t.isalnum() or t in string.punctuation or
                 t == " " or t == '…' or t == '“'):
                   error_token_flag = True
                   break
                
            if error_token_flag:
                self.error_indexes.append((decode_token_text, token_id))
        print(f"Error subtoken amount: {len(self.error_indexes)}")
        
    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        for _, token_id in self.error_indexes:
            logits[token_id] = float('-inf')
        return logits


class GenerationContext:
    def __init__(self, token_ids, generated_text, max_op_bracket):
        self.token_ids = token_ids
        self.generated_text = generated_text
        print(f"generated_text: {self.generated_text}")
        self.op_amount = self.generated_text.count("[")
        self.end_amount = self.generated_text.count("]")
        print(f"op_amount: {self.op_amount}, end_amount: {self.end_amount}")
        self.max_op_bracket = max_op_bracket

class BracketLogitsProcessor:
    def __init__(self, tokenizer, op_code, end_code, eos_id, first_root=False):
        self.max_op_bracket = None
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        partial_bracket_codes = [(k, v) for k, v in vocab.items() if "[" in k or "]" in k]

        self.force_first_constraints = ForceFirstTokenConstraint(op_code)
        self.force_root_constraints = ForceRootPrefixConstraint(first_root, self.tokenizer)
        self.force_finish_constraints = ForceFinishConstraint(eos_id)
        self.force_end_constraints = ForceEndConstraint(end_code)
        
        self.restrict_bracket_after_open_constraints = RestrictBracketAfterOpenConstraint(partial_bracket_codes)
        self.restrict_balance_constraints = RestrictBalanceBracketConstraint(partial_bracket_codes)
        self.restrict_open_constraints = RestrictOpenConstraint(partial_bracket_codes)
        self.restrict_error_constraints = RestrictErrorTokenConstraint(partial_bracket_codes, self.tokenizer)

    def set_max_op_bracket(self, max_op_bracket): # TODO: в контекст, т.к. используется в ForceClosingConstraint, а его нельзя создавать до set_max_op_bracket
        self.max_op_bracket = max_op_bracket * 2

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, token_ids, logits):
        print(token_ids)

        generated_text = self.tokenizer.decode(token_ids)
        context = GenerationContext(token_ids, generated_text, self.max_op_bracket)
        
        logits = logits.clone()
        if self.force_first_constraints.check(context):
            logits = self.force_first_constraints(logits, context)
        elif self.force_root_constraints.check(context):
            logits = self.force_root_constraints(logits, context)
        elif self.force_finish_constraints.check(context):
            logits = self.force_finish_constraints(logits, context)
        elif self.force_end_constraints.check(context):
            logits = self.force_end_constraints(logits, context)
        else:
            if self.restrict_error_constraints.check(context):
                logits = self.restrict_error_constraints(logits, context)
            if self.restrict_open_constraints.check(context):
                logits = self.restrict_open_constraints(logits, context)
            if self.restrict_bracket_after_open_constraints.check(context):
                logits = self.restrict_bracket_after_open_constraints(logits, context)
            logits = self.restrict_balance_constraints(logits, context)
        print()
        return logits


def create_logit_processor(logit_params, tokenizer):
    vocab = tokenizer.get_vocab()
    #print(bracket_tokens)

    op_code = tokenizer.encode("[")
    assert len(op_code) == 1
    op_code = op_code[0]
    end_code = tokenizer.encode("]")
    assert len(end_code) == 1
    end_code = end_code[0]

    eos_code = 2 # TODO: Сделать константу, используется в других местах

    #print(op_code, end_code, eos_code)

    logit_processor = BracketLogitsProcessor(tokenizer, op_code, end_code, eos_code)
    return logit_processor
