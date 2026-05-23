from abc import ABC, abstractmethod
import re
import string
from typing import List, Tuple

import torch


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
    
    def __init__(self, partial_bracket_codes):
        self.partial_bracket_codes = partial_bracket_codes
        
    def check(self, context):
    # First subtoken has to be "["
        return len(context.token_ids) == 0
    
    def __call__(self, logits, context):
        print("Forcing first [")
        codes_with_logits = [(token_text, token_id, float(logits[token_id]))
                             for token_text, token_id in self.partial_bracket_codes]
        logits[:] = -torch.inf

        for token_text, token_id, token_logit in codes_with_logits:
            if token_text[0] == "[":
                logits[token_id] = token_logit
        return logits
        
class ForceRootPrefixConstraint(Constraint):
     def __init__(self, applying_first_root, tokenizer):
         self.applying_first_root = applying_first_root
         self.root_prefix = "[root["
         self.tokenizer = tokenizer

     def check(self, context):
         return self.applying_first_root and len(context.generated_text) <= 6
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
    
    def __init__(self, partial_bracket_codes, applying_max_amount):
        self.partial_bracket_codes = partial_bracket_codes
        self.applying_max_amount = applying_max_amount
        
    def check(self, context):
        if not self.applying_max_amount:
            return False
        return context.check_all_open() and \
            not context.check_all_end() and \
            context.generated_text[-1] == "]"
        # Generate some last "]"

    def __call__(self, logits, context):
        bracket_diff = context.op_amount - context.end_amount
        print("Forcing last ]")
        codes_with_logits = [(token_text, token_id, float(logits[token_id]))
                             for token_text, token_id in self.partial_bracket_codes]
        logits[:] = -torch.inf
        for token_text, token_id, token_logit in codes_with_logits:
            if set(token_text) == {"]"} and bracket_diff - len(token_text) >= 0:
                logits[token_id] = token_logit
        return logits

class ForceFinishConstraint(Constraint):
    """"""
    def __init__(self, eos_id, applying_max_amount, soft_max_amount):
        self.eos_id = eos_id
        self.applying_max_amount = applying_max_amount
        self.soft_max_amount = soft_max_amount
    
    def check(self, context):
        if not self.applying_max_amount:
            # Ограничений на количество скобок нет
            if context.op_amount == context.end_amount:
            # Сгенерирована законченная скобочная последовательность, дальше нельзя генерировать
                return True
            return False
        if self.soft_max_amount: # TODO: проверить логику
            if context.op_amount == context.end_amount:
            # Сгенерирована законченная скобочная последовательность, дальше нельзя генерировать
                return True
            return False
        return context.check_all_open() and context.check_all_end() # Finish generating
    
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
        
class RestrictTextAfterEndConstraint(Constraint):
    def __init__(self, partial_bracket_codes, eos_ids):
        self.partial_bracket_codes = partial_bracket_codes
        self.eos_ids = eos_ids

    def check(self, context):
        return context.generated_text[-1] == "]"
        
    def __call__(self, logits, context): # TODO: запрет нужен для всех!!!
        mask_tensor = torch.full(logits.shape, -float('inf'), device=logits.device)
        for token_text, token_id in self.partial_bracket_codes:
            if token_text[0] == "]" or token_text[0] == "[":
                mask_tensor[token_id] = 0
        for token_id in self.eos_ids:
            mask_tensor[token_id] = 0
        logits += mask_tensor
        return logits

        
class RestrictBalanceBracketConstraint(Constraint):
    """"""
    
    def __init__(self, partial_bracket_codes, applying_max_amount, soft_max_amount):
        self.partial_bracket_codes = partial_bracket_codes
        self.applying_max_amount = applying_max_amount
        self.soft_max_amount = soft_max_amount
        
    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        bracket_diff = context.op_amount - context.end_amount
        for token_text, token_id in self.partial_bracket_codes:
            inf_flag = False
            for i in range(1, len(token_text)):
                token_slice = token_text[:i]
                if bracket_diff + token_slice.count("[") - token_slice.count("]") < 1:
                    # Ошибка вида "[Остается][" при добавлении "]["
                    inf_flag = True
            if not inf_flag:
                new_bracket_diff = bracket_diff + \
                    token_text.count("[") - token_text.count("]")
                if self.applying_max_amount and not self.soft_max_amount: # Есть жесткое ограничение по количеству открытых скобок
                    if not context.check_all_open(): # И не все [ сгенерированы
                        # То есть нельзя делать diff = 0
                        if new_bracket_diff < 1:
                            inf_flag = True
                    else: # Все [ сгенерированы, можно diff = 0
                        if new_bracket_diff < 0:
                            inf_flag = True
                else:
                # Ограничений по количеству скобок нет, можно уходить в ноль (тогда потом будет force end)
                    if new_bracket_diff < 0:
                        inf_flag = True
            if inf_flag:
                logits[token_id] = float('-inf')
        return logits

class RestrictOpenConstraint(Constraint):
    """"""
    
    def __init__(self, partial_bracket_codes, applying_max_amount):
        self.partial_bracket_codes = partial_bracket_codes
        self.applying_max_amount = applying_max_amount
        
    def check(self, context):
        if not self.applying_max_amount:
            return False
        return context.check_all_open() and not context.check_all_end()
    
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
        vocab = tokenizer.get_vocab()
        for token_text, token_id in vocab.items():
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
        for token_text in ["<think>", "</think>", "<|im_start|>"]:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
            self.error_indexes.append((token_text, token_id))

        print(f"Error subtoken amount: {len(self.error_indexes)}")
        
    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        for _, token_id in self.error_indexes:
            logits[token_id] = float('-inf')
        return logits


class RestrictUnbalancedEOSConstraint(Constraint):
    """"""
    
    def __init__(self, eos_ids):
        self.eos_ids = eos_ids
        
    def check(self, context):
        return context.op_amount != context.end_amount
    
    def __call__(self, logits, context):
        print("Restriction for eos (because of unbalancing)")
        for eos_id in self.eos_ids:
            logits[eos_id] = -torch.inf
        return logits

def fold_bracket_seq(s_param):
    s = s_param[:]
    s = s.replace(" ", "")
    s = re.sub(r'[^\[\]TWC]+', lambda m: 'T', s)
    #print(s)
    while "TT" in s:
        s = s.replace("TT", "T")
    s = re.sub(r'\[T\]', lambda m: 'W', s)
    #print(s)

    prev_s = ''
    while prev_s != s:
        prev_s = s
        s = re.sub(r'\[TC*WC*\]', lambda m: 'C', s)
        #print(s, prev_s)
        s = re.sub(r'\[TC*WC*WC*\]', lambda m: 'E', s)
        s = re.sub(r'\[TC*\]', lambda m: 'E', s)
        s = re.sub(r'\[[W|C]*\]', lambda m: 'E', s)
        s = re.sub(r'\[[W|C]+T[W|C]*]\]', lambda m: 'E', s)
        s = re.sub(r'\[[W|C]*T[W|C]*T[W|C]*]\]', lambda m: 'E', s)

        if 'E' in s:
          break
    #print("fold_bracket_seq", s_param, s)
    return s


class RestrictUncorrectLevelConstraint(Constraint):

    def __init__(self, partial_bracket_codes):
        self.partial_bracket_codes = partial_bracket_codes

    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        print("Restrictions for grct levels")
        for token_text, token_id in self.partial_bracket_codes:
            if logits[token_id] != -torch.inf:
                if "E" in fold_bracket_seq(context.re_text + token_text.lower()):
                    logits[token_id] = -torch.inf
                    print(f"Restriction for {token_id}")
        return logits


class GenerationContext:
    def __init__(self, token_ids, generated_text, max_op_bracket,
            last_processed_text, last_processed_re):
        self.token_ids = token_ids
        self.generated_text = generated_text
        print(f"generated_text: {self.generated_text}")
        self.op_amount = self.generated_text.count("[")
        self.end_amount = self.generated_text.count("]")
        print(f"op_amount: {self.op_amount}, end_amount: {self.end_amount}")
        self.max_op_bracket = max_op_bracket
        if last_processed_text is None:
            last_processed_text = ""
            last_processed_re = ""
        new_text = self.generated_text[len(last_processed_text):]
        self.re_text = fold_bracket_seq(last_processed_re + new_text.lower()) # TODO: Сделать отдельный класс с хранением re и добавлением нового с lower)
        print(self.re_text)

    def check_all_open(self):
        return self.op_amount == self.max_op_bracket
    
    def check_all_end(self):
        return self.end_amount == self.max_op_bracket

class OriginalLogitsProcessor:
    def __init__(self, tokenizer, logit_params):
        self.logit_params = logit_params
        self.max_op_bracket = None

    def create_new_context(self, max_op_bracket): 
        self.max_op_bracket = max_op_bracket

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, token_ids, logits):
        return logits

class BracketLogitsProcessor:
    def __init__(self, tokenizer, logit_params):
        optional_constraints = logit_params.get("optional_constraints", set())
        self.max_op_bracket = None
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        partial_bracket_codes = [(k, v) for k, v in vocab.items() if "[" in k or "]" in k]

        applying_first_root = "root" in optional_constraints
        print(f"applying_first_root: {applying_first_root}")
        applying_max_amount = "max_amount" in optional_constraints
        print(f"applying_max_amount: {applying_max_amount}") # Ровно заданное количество [
        soft_max_amount = "soft_max_amount" in optional_constraints
        print(f"soft_max_amount: {soft_max_amount}") # Не более заданного количества [
        if soft_max_amount:
            applying_max_amount = True

        self.force_first_constraints = ForceFirstTokenConstraint(partial_bracket_codes)
        self.force_root_constraints = ForceRootPrefixConstraint(applying_first_root, self.tokenizer)
        self.force_finish_constraints = ForceFinishConstraint(tokenizer.eos_token_id,
            applying_max_amount, soft_max_amount)
        self.force_end_constraints = ForceEndConstraint(partial_bracket_codes, applying_max_amount)
        
        eos_ids = [tokenizer.old_eos_token_id, tokenizer.eos_token_id]
        print(f"eos_ids: {eos_ids}")
        
        self.restrict_bracket_after_open_constraints = RestrictBracketAfterOpenConstraint(partial_bracket_codes)
        self.restrict_text_after_end_constraints = RestrictTextAfterEndConstraint(partial_bracket_codes, eos_ids)
        self.restrict_balance_constraints = RestrictBalanceBracketConstraint(partial_bracket_codes,
            applying_max_amount, soft_max_amount)
        self.restrict_open_constraints = RestrictOpenConstraint(partial_bracket_codes, applying_max_amount)
        self.restrict_error_constraints = RestrictErrorTokenConstraint(partial_bracket_codes, self.tokenizer)
        self.restrict_unbalanced_eos_constraints = RestrictUnbalancedEOSConstraint(eos_ids)

        self.restrict_uncorrect_level_constraints = RestrictUncorrectLevelConstraint(partial_bracket_codes)

        self.mul_coeff = logit_params.get("mul_coeff", 1)
        self.add_coeff = logit_params.get("add_coeff", 0)

        self.last_processed_text = None
        self.last_processed_re = None


    def create_new_context(self, max_op_bracket): 
        self.max_op_bracket = (self.mul_coeff * max_op_bracket + \
            self.add_coeff) * 2
        self.last_processed_text = None
        self.last_processed_re = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, token_ids, logits):
        print(token_ids)

        generated_text = self.tokenizer.decode(token_ids)
        context = GenerationContext(token_ids, generated_text, self.max_op_bracket,
                    self.last_processed_text, self.last_processed_re)
        # max_op_bracket в контекст, т.к. используется в ForceClosingConstraint,
        # а его нельзя создавать до create_new_context
        self.last_processed_text = context.generated_text
        self.last_processed_re = context.re_text
        
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
            if self.restrict_text_after_end_constraints.check(context):
                logits = self.restrict_text_after_end_constraints(logits, context)
            if self.restrict_unbalanced_eos_constraints.check(context):
                logits = self.restrict_unbalanced_eos_constraints(logits, context)
            if self.restrict_balance_constraints.check(context):
                logits = self.restrict_balance_constraints(logits, context)
            if self.restrict_uncorrect_level_constraints.check(context):
                logits = self.restrict_uncorrect_level_constraints(logits, context)
        print()
        return logits


def create_logit_processor(logit_params, tokenizer):
    if logit_params.get("name", "") == "original_logits":
        logit_processor = OriginalLogitsProcessor(tokenizer, logit_params)
    else:
        logit_processor = BracketLogitsProcessor(tokenizer, logit_params)
    return logit_processor
