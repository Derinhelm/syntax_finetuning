from abc import ABC, abstractmethod
import re
import string
from typing import List, Tuple
from collections import Counter


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
# Prefix constraints
RUSSIAN_RELATIONS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc',
             'ccomp', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det',
             'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'iobj', 'list',
             'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan',
             'parataxis', 'punct', 'vocative', 'xcomp'] # 'root'

ENGLISH_RELATIONS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
            'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det',
            'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 
            'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl',
            'orphan', 'parataxis', 'punct', 'reparandum', 'vocative', 'xcomp'] # 'root'

class PrefixGenerator:    
    def __init__(self):
        self.name = "prefix"        

    def _get_last_level(self, context):
        last_re_level_ind = context.re_text.rfind("[|T")
        if last_re_level_ind == -1:
            last_re_level_ind = 0
        return context.re_text[last_re_level_ind:]

    def _create_relation_prefixes(self, context):
        #print("{context.re_text.count('[|T')=}", f"{context.re_text.count('|W')=}",
        #    f"{context.last_unused_tokens.total()=}")
        #print(context.re_text.count("[|T") - context.re_text.count("|W") == \
        #    context.last_unused_tokens.total()) 
        if context.re_text.count("[|T") - context.re_text.count("|W") == \
                context.last_unused_tokens.total():
            # [|T[|T[ - если 2 last_unused_tokens, нельзя генерировать relations (нечем закрыть)
            # [|T[|T[|W] - если 2 last_unused_tokens, можно генерировать relations
            # [|T[|W][|T[ - если 2 last_unused_tokens, можно генерировать relations
            return []
        return ENGLISH_RELATIONS

    def _create_form_prefixes(self, context):
        last_re_level = self._get_last_level(context)
        if "|W" in last_re_level:
            return []
        else:
            return context.last_unused_tokens

    def __call__(self, context):
        #print(f"{self._create_form_prefixes(context)}")
        #print(f"{self._create_relation_prefixes(context)}")
        generated_text = context.generated_text
        root_text = "[root["
        if generated_text == "":
            return [root_text]
        elif root_text.startswith(generated_text):
            return [root_text[len(generated_text):]]
        elif generated_text[-1] == "]":
            if context.op_amount == context.end_amount:
                return ["eos"]
            return ["]"] + ["[" + el + "]" for el in
                self._create_form_prefixes(context)] + \
                ["[" + el + "[" for el in
                self._create_relation_prefixes(context)]
        elif context.last_unused_tokens.total() == 0:
            return ["]"]
        elif generated_text[-1] == "[":
            return [el + "]" for el in
                self._create_form_prefixes(context)] + \
                [el + "[" for el in
                self._create_relation_prefixes(context)] 
        else:
            last_el_text = generated_text.split("[")[-1]
            s_prefixes = [el[len(last_el_text):] + "]" for el in
                self._create_form_prefixes(context)
                if el.startswith(last_el_text)] + \
                [el[len(last_el_text):] + "[" for el in
                self._create_relation_prefixes(context)
                if el.startswith(last_el_text)]
            if "[" in s_prefixes: # TODO: укорить
                s_prefixes.remove("[")
                s_prefixes += ["[" + el + "]" for el in
                    self._create_form_prefixes(context)] + \
                        ["[" + el + "[" for el in
                        self._create_relation_prefixes(context)] 
            # TODO: if "]" in s_prefixes: - ввести "]" + конец ?
            #    s_prefixes.remove("]")
            return s_prefixes


from genlm.backend.tokenization import decode_vocab
import marisa_trie

class PrefixFinder:
    def __init__(self, tokenizer):
        super().__init__()
        byte_vocab, _ = decode_vocab(tokenizer)
        byte_subtokens = [subtoken.byte_string for subtoken in byte_vocab]
        self.bytes_to_id = {subtoken.byte_string: subtoken.token_id
                            for subtoken in byte_vocab}
        self.trie = marisa_trie.BinaryTrie(byte_subtokens)

    def __call__(self, prefixes: str):
        allow_ids = []
        for target_str in prefixes:
            target = target_str.encode("utf-8")
            # Условие 1: токены, которые являются префиксом target
            target_prefixes = list(self.trie.iter_prefixes(target))
            #print(f"{target_prefixes=}")
            # Условие 2: токены, для которых target — префикс
            target_continuations = list(self.trie.iterkeys(target))
            #print(f"{target_continuations=}")
            allow_ids += [self.bytes_to_id[b] \
                for b in target_prefixes + target_continuations]
            #print(f"{allow_ids=}")

        return list(set(allow_ids))

class PrefixConstraint(Constraint):
    def __init__(self, tokenizer, eos_ids):
        self.prefix_generator = PrefixGenerator()
        self.prefix_checker = PrefixFinder(tokenizer)
        self.eos_ids = eos_ids

    def check(self, context):
        return True

    def __call__(self, logits, context):
        prefixes = self.prefix_generator(context)
        if prefixes == ["eos"]:
            allow_ids = self.eos_ids
        else:
            allow_ids = self.prefix_checker(prefixes)
        inf_mask = torch.ones_like(logits, dtype=torch.bool)
        inf_mask[allow_ids] = False
        logits[inf_mask] = -torch.inf
        return logits

# =============================================
# Force constraints

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
            len(context.generated_text) > 0 and \
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
        if context.op_amount == 0:
            return False
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
        return len(context.generated_text) > 0 and \
            context.generated_text[-1] == "["
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
        return len(context.generated_text) > 0 and \
            context.generated_text[-1] == "]"
        
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
                self.error_indexes.append(token_id)
        for token_text in ["<think>", "</think>", "<|im_start|>"]:
            token_id = tokenizer.convert_tokens_to_ids(token_text)
            self.error_indexes.append(token_id)

        print(f"Error subtoken amount: {len(self.error_indexes)}")
        
    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        logits[self.error_indexes] = float('-inf')
        return logits


class RestrictUnbalancedEOSConstraint(Constraint):
    """"""
    
    def __init__(self, eos_ids):
        self.eos_ids = eos_ids
        
    def check(self, context):
        return context.op_amount > 0 and \
            context.op_amount != context.end_amount
    
    def __call__(self, logits, context):
        print("Restriction for eos (because of unbalancing)")
        logits[self.eos_ids] = -torch.inf
        return logits

def constant_check(x):
    return "[" not in x[1:-1] and "]" not in x[1:-1] and \
        "|T" not in x and "|C" not in x and "|W" not in x and "|E" not in x

def change_text(change_fun, right_border_symbol, s):
    #print(f"{right_border_symbol=}")
    ready_border = 0
    left_border_in_fragment = s[ready_border:].find("[")
    right_border_in_fragment = s[ready_border + left_border_in_fragment + 1:].find(right_border_symbol)
    while left_border_in_fragment != -1 and right_border_in_fragment != -1:
        left_border = ready_border + left_border_in_fragment
        change_text = change_fun(s[left_border:left_border + 1 + right_border_in_fragment + 1])
        #print(f"{left_border=} {right_border_in_fragment=} {s[left_border:left_border + 1 + right_border_in_fragment + 1]} {change_text}")
        new_s = s[:left_border] + change_text + s[left_border + 1 + right_border_in_fragment + 1:]
        if "|E" in change_text:
            return new_s
        ready_border = left_border + 1 # Все, что левее left_border - неизменно
        s = new_s
        #print(f"start with ready_border: {s[ready_border:]} {s[ready_border:].find('[')}")
        left_border_in_fragment = s[ready_border:].find("[")
        if left_border_in_fragment == -1:
            break
        #print(f"{left_border_in_fragment=}")
        #print(f"{s[ready_border + left_border_in_fragment + 1:]} {s[ready_border + left_border_in_fragment + 1:].find(right_border_symbol)}")
        right_border_in_fragment = s[ready_border + left_border_in_fragment + 1:].find(right_border_symbol)
        #print(f"{right_border_in_fragment=}")
    return s

def fold_bracket_seq(s, allow_relations=None, unused_tokens=None, drop_tokens=False):
    # >>> fold_bracket_seq("[roet[")
    # '|E'

    s = s.replace(" ", "")
   
    if s[:6] == "[root[":
        s = "[|T[" + s[6:]
    elif "[root[".startswith(s):
        return s
    elif not s.startswith("[|T"):
        return "|E"
    #print(1, s)
    # >>> fold_bracket_seq('[root[К', {'nmod'}, Counter(['Дым', 'Дом']))
    # '[|T[|E'
    last_op_bracket = s.rfind("[")
    last_text = s[last_op_bracket + 1:]
    # >>> fold_bracket_seq('[root[Дом]]', {'nmod'}, Counter(['Дым', 'Дом'])) - |
    if not last_text.startswith("|T") and \
        "]" not in last_text and unused_tokens is not None and \
            (len([token for token in unused_tokens if token.startswith(last_text)]) == 0 and
            (len([rel for rel in allow_relations if rel.startswith(last_text)]) == 0)):
        return s[:last_op_bracket + 1] + "|E"
    if last_text.count("|W") > 1:
        return s[:last_op_bracket + 1] + "|E"
    if allow_relations is not None:
        f_rel = lambda x: ('[|T[' if x[1:-1] in allow_relations else '[|E[') if constant_check(x) else x

    else:
        f_rel = lambda x: '[|T[' if constant_check(x) else x

    if unused_tokens is not None:
        if drop_tokens:
            def f_form_drop(x):
                if not constant_check(x):
                    return x
                form_text = x[1:-1]
                if form_text in unused_tokens and unused_tokens[form_text] > 0:
                    unused_tokens[form_text] -= 1
                    if unused_tokens[form_text] <= 0:
                        unused_tokens.pop(form_text)
                    return "[|T]"
                else:
                    return "[|E]"
            f_form = f_form_drop
        else:
            f_form = lambda x: ('[|T]' if x[1:-1] in unused_tokens else '[|E]') if constant_check(x) else x
    else:
        f_form = lambda x: '[|T]' if constant_check(x) else x

    # >>> fold_bracket_seq('[root[Дом][nmod[Тен]]]', {'nmod'}, Counter(['Дым', 'Дом', 'Тен'])) - |C
    s = change_text(f_form, "]", s)
    if "|E" in s:
        return s
    s = change_text(f_rel, "[", s)
    if '|E' in s:
        return s

    f_check_level_two_words = lambda x: "|E" if x.count("|W") > 1 else x 
    # Против [|T|C|W|W[
    s = change_text(f_check_level_two_words, "[", s)

    if '|E' in s:
        return s
    #print("after text", s)

    if unused_tokens is not None:
        can_open = unused_tokens.total()
        if last_text in unused_tokens:
            can_open -= 1
        if not drop_tokens and "[|T]" in s:
            can_open -= s.count("[|T]")
        # если в предложении одно слово, нельзя [|T[|W[. Любая [ требует слова для "разрешения"
        if can_open == 0 and s[-1] == "[":
            return "|E" # TODO: более понятную строку
    
    s = re.sub(r'\[\|T\]', '|W', s)

    while '|E' not in s and "]" in s and s[0] == "[":
        end_bracket = s.find("]")
        op_bracket = s[:end_bracket].rfind("[")
        if s[op_bracket + 1:op_bracket + 3] != "|T" or s[op_bracket + 2: end_bracket].count("|W") != 1:
            s = s[:op_bracket + 1] + "|E" + s[end_bracket:]
            break
        else:
            s = s[:op_bracket] + "|C" + s[end_bracket + 1:]

    if unused_tokens is not None and s.count("[|T") - s.count("|W") == unused_tokens.total():
        # слов осталось ровно столько, сколько нужно, чтобы закрыть все открытые |T
        # Нельзя открывать новые уровни, только новые слова
        last_level_start = s.rfind("[|T")
        if last_level_start != -1:
            last_level_text = s[last_level_start:]
            if "|W" in last_level_text and last_level_text.count("[") > 1:
                return "E"
    if s != "" and s[0] != "[" and s != "|C":
        s = "|E"
    return s


class RestrictUncorrectLevelConstraint(Constraint):

    def __init__(self, partial_bracket_codes, tokenizer):
        self.partial_bracket_codes = partial_bracket_codes
        self.tokenizer = tokenizer

    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        print("Restrictions for grct levels")
        #for token_text, token_id in self.partial_bracket_codes: # TODO: Только для оставшихся разрешенными
            # TODO: Куда ставить проверку после проверки по префиксам ?

        valid_indexes = []
        iter_i = 0
        max_logits = None
        MIN_VALUE_LEN = 3 # TODO: больше него нельзя делать выбор элементов
        ITER_SIZE = 10
        while len(valid_indexes) < MIN_VALUE_LEN and (max_logits is None or not max_logits[-1].isinf()):
            max_logits, max_indices = torch.topk(logits, k = ITER_SIZE * (iter_i + 1))
            max_token_texts = [self.tokenizer.decode(ind) for ind in max_indices]
            #print(f"{max_logits=}")
            #print(f"{max_indices=}")
            #print(f"{max_token_texts=}")
            #print(f"{iter_i=}")
            for token_i, token_id in enumerate(max_indices[ITER_SIZE * iter_i:]):
                if max_logits[token_i + ITER_SIZE * iter_i].isinf():
                    break
                token_text = max_token_texts[token_i + ITER_SIZE * iter_i]
                fold_res = fold_bracket_seq(context.re_text + token_text,
                            ENGLISH_RELATIONS, context.last_unused_tokens, False)
                #print(f"{fold_res=}")
                if "|E" not in fold_res:
                        # TODO: Проверка с учетом типа связи/формы и без них
                        # Проверка, без изменения набора токенов
                    valid_indexes.append(token_id.item())
            iter_i += 1
            #print(f"{valid_indexes=}")
        valid_mask = torch.zeros_like(logits, dtype=torch.bool)
        valid_mask[valid_indexes] = True
        logits[~valid_mask] = float('-inf')
        return logits


class GenerationContext:
    def __init__(self, token_ids, generated_text, max_op_bracket,
            last_processed_text, last_processed_re,
            last_unused_tokens):
        self.token_ids = token_ids
        self.generated_text = generated_text
        self.op_amount = self.generated_text.count("[")
        self.end_amount = self.generated_text.count("]")
        self.max_op_bracket = max_op_bracket
        if last_processed_text is None:
            last_processed_text = ""
            last_processed_re = ""
        new_text = self.generated_text[len(last_processed_text):]
        self.re_text = fold_bracket_seq(last_processed_re + new_text, # TODO: не будет работать для строк с |, в SynTagRus нет
            ENGLISH_RELATIONS, last_unused_tokens, True)
        # TODO: Сделать отдельный класс с хранением re и добавлением нового с lower)
        self.last_unused_tokens = last_unused_tokens
        

    def check_all_open(self):
        return self.op_amount == self.max_op_bracket
    
    def check_all_end(self):
        return self.end_amount == self.max_op_bracket

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

        self.force_finish_constraints = ForceFinishConstraint(tokenizer.eos_token_id,
            applying_max_amount, soft_max_amount)
        self.force_end_constraints = ForceEndConstraint(partial_bracket_codes, applying_max_amount)

        eos_ids = [tokenizer.old_eos_token_id, tokenizer.eos_token_id]
        print(f"eos_ids: {eos_ids}")

        self.prefix_constraints = PrefixConstraint(tokenizer, eos_ids)

        self.restrict_bracket_after_open_constraints = RestrictBracketAfterOpenConstraint(partial_bracket_codes)
        self.restrict_text_after_end_constraints = RestrictTextAfterEndConstraint(partial_bracket_codes, eos_ids)
        self.restrict_balance_constraints = RestrictBalanceBracketConstraint(partial_bracket_codes,
            applying_max_amount, soft_max_amount)
        self.restrict_open_constraints = RestrictOpenConstraint(partial_bracket_codes, applying_max_amount)
        self.restrict_error_constraints = RestrictErrorTokenConstraint(partial_bracket_codes, self.tokenizer)
        self.restrict_unbalanced_eos_constraints = RestrictUnbalancedEOSConstraint(eos_ids)

        self.restrict_uncorrect_level_constraints = RestrictUncorrectLevelConstraint(
            partial_bracket_codes, self.tokenizer)

        self.mul_coeff = logit_params.get("mul_coeff", 1)
        self.add_coeff = logit_params.get("add_coeff", 0)

        self.last_processed_text = None
        self.last_processed_re = None
        self.last_unused_tokens = None


    def create_new_context(self, input_tokens): 
        self.max_op_bracket = (self.mul_coeff * len(input_tokens) + \
            self.add_coeff) * 2
        self.last_processed_text = None
        self.last_processed_re = None
        self.last_unused_tokens = Counter(input_tokens)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, token_ids, logits):
        #import time
        #time_list = []
        #torch.cuda.synchronize()
        #ts_all = time.perf_counter()
        #torch.cuda.synchronize()
        #ts = time.perf_counter()
        #torch.cuda.synchronize()
        #tf = time.perf_counter()
        #time_list.append(("print", tf - ts))

        #torch.cuda.synchronize()
        #ts = time.perf_counter()
        generated_text = self.tokenizer.decode(token_ids)
        #torch.cuda.synchronize()

        #tf = time.perf_counter()
        #time_list.append(("decode", tf - ts))

        #torch.cuda.synchronize()
        #ts = time.perf_counter()
        context = GenerationContext(token_ids, generated_text, self.max_op_bracket,
                    self.last_processed_text, self.last_processed_re,
                    self.last_unused_tokens)
        print(f"{context.__dict__=}")
        # max_op_bracket в контекст, т.к. используется в ForceClosingConstraint,
        # а его нельзя создавать до create_new_context
        self.last_processed_text = context.generated_text
        self.last_processed_re = context.re_text
        self.last_unused_tokens = context.last_unused_tokens
        #torch.cuda.synchronize()
        #tf = time.perf_counter()
        #time_list.append(("context", tf - ts))

        #torch.cuda.synchronize()
        #ts = time.perf_counter()
        logits = logits.clone()
        #torch.cuda.synchronize()
        #tf = time.perf_counter()
        #time_list.append(('clone', tf - ts))

        if self.force_finish_constraints.check(context):
            logits = self.force_finish_constraints(logits, context)
        elif self.force_end_constraints.check(context):
            logits = self.force_end_constraints(logits, context)
        else:
            #torch.cuda.synchronize()
            #ts = time.perf_counter()
            if self.prefix_constraints.check(context):
                logits = self.prefix_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("prefix", tf - ts))

            #torch.cuda.synchronize()
            #ts = time.perf_counter()
            if self.restrict_error_constraints.check(context):
                logits = self.restrict_error_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("error", tf - ts))

            #torch.cuda.synchronize()
            #ts = time.perf_counter()
            if self.restrict_open_constraints.check(context):
                logits = self.restrict_open_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("open", tf - ts))

            #torch.cuda.synchronize()
            #ts = time.perf_counter()
            if self.restrict_bracket_after_open_constraints.check(context):
                logits = self.restrict_bracket_after_open_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("brack-after-open", tf - ts))

            #torch.cuda.synchronize()
            #ts = time.perf_counter()
            if self.restrict_text_after_end_constraints.check(context):
                logits = self.restrict_text_after_end_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("text-after-close", tf - ts))

            #torch.cuda.synchronize()
            #ts = time.perf_counter()
            if self.restrict_unbalanced_eos_constraints.check(context):
                logits = self.restrict_unbalanced_eos_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("unbal", tf - ts))

            #torch.cuda.synchronize()
            #ts = time.perf_counter()
            if self.restrict_balance_constraints.check(context):
                logits = self.restrict_balance_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("balance", tf - ts))

            #torch.cuda.synchronize()
            #ts = time.perf_counter()          
            if self.restrict_uncorrect_level_constraints.check(context):
                logits = self.restrict_uncorrect_level_constraints(logits, context)
            #torch.cuda.synchronize()
            #tf = time.perf_counter()
            #time_list.append(("levels", tf - ts))
        #torch.cuda.synchronize()
        #tf_all = time.perf_counter()
        #print(tf_all - ts_all)
        #print(time_list)
        return logits


def create_logit_processor(logit_params, tokenizer):
    if logit_params.get("name", "") == "original_logits":
        logit_processor = OriginalLogitsProcessor(tokenizer, logit_params)
    else:
        logit_processor = BracketLogitsProcessor(tokenizer, logit_params)
    return logit_processor
