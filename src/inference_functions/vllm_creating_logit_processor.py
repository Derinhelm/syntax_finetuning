from abc import ABC, abstractmethod
import copy
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

CHINESE_RELATIONS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc',
           'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'det', 'discourse',
           'dislocated', 'flat', 'iobj', 'mark', 'nmod', 'nsubj', 'nummod', 'obj',
           'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'vocative', 'xcomp'] # 'root', GSD

FINNISH_RELATIONS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux',
                     'case', 'cc', 'ccomp', 'compound', 'conj', 'cop',
                     'csubj', 'dep', 'det', 'discourse', 'expl',
                     'fixed', 'flat', 'goeswith', 'mark', 'nmod',
                     'nsubj', 'nummod', 'obj', 'obl', 'orphan',
                     'punct', 'reparandum', 'vocative', 'xcomp'] # 'root', FTB

FRENCH_RELATIONS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux',
                    'case', 'cc', 'ccomp', 'compound', 'conj', 'cop',
                    'csubj', 'dep', 'det', 'discourse', 'dislocated',
                    'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'mark',
                    'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan',
                    'parataxis', 'punct', 'vocative', 'xcomp'] # 'root', GSD

JAPANESE_RELATIONS = ['acl', 'advcl', 'advmod', 'amod', 'aux', 'case',
                      'cc', 'ccomp', 'compound', 'cop', 'csubj', 'dep',
                      'det', 'discourse', 'fixed', 'iobj', 'mark', 'nmod',
                      'nsubj', 'nummod', 'obj', 'obl', 'punct'] # 'root', GSD

RELATION_DICT = {None: RUSSIAN_RELATIONS, "ru": RUSSIAN_RELATIONS,
                 "eng": ENGLISH_RELATIONS, "ch": CHINESE_RELATIONS,
                 "fi": FINNISH_RELATIONS, "fr": FRENCH_RELATIONS,
                 "ja": JAPANESE_RELATIONS}

OPEN_BYTE = b"["[0]
END_BYTE = b"]"[0]

class PrefixGenerator:    
    def __init__(self, relations):
        self.name = "prefix"
        self.relations = relations        

    def _create_relation_prefixes(self, context):
        if context.re_before_right_open.count(b"[|T") - context.re_before_right_open.count(b"|W") == \
                context.last_unused_tokens.total():
            # [|T[|T[ - если 2 last_unused_tokens, нельзя генерировать relations (нечем закрыть)
            # [|T[|T[|W] - если 2 last_unused_tokens, можно генерировать relations
            # [|T[|W][|T[ - если 2 last_unused_tokens, можно генерировать relations
            return []
        return self.relations

    def _create_form_prefixes(self, context):
        # '[|T|W[|T|C|W|C[nmod' - [nmod - это не форма
        # '[|T|W[|T|C|W|C' - [|T|C|W|C
        # '[|T|W[|T|C|W|C[nmod' - можно добавить [Дом (если ок по количеству)

        # TODO1
        if context.bytes_after_right_open != b"": # TODO: зачем ?
            if b"]" not in context.bytes_after_right_open and context.bytes_after_right_open in self.relations:
                return context.last_unused_tokens # Нарушается логика для некоторых вариантов. Расписать

        last_re_level_ind = context.re_before_right_open.rfind(b"[|T")
        if last_re_level_ind == -1:
            last_re_level_ind = 0
        if b"|W" in context.re_before_right_open[last_re_level_ind:]:
            return []
        else:
            return context.last_unused_tokens

    def __call__(self, context):
        #print(f"{self._create_form_prefixes(context)}")
        #print(f"{self._create_relation_prefixes(context)}")
        root_bytes = b"[root["
        if context.generated_bytes == b"":
            return [root_bytes]
        elif root_bytes.startswith(context.generated_bytes) and root_bytes != context.generated_bytes:
            return [root_bytes[len(context.generated_bytes):]]
        elif context.generated_bytes.endswith(b"]"):
            if context.op_amount == context.end_amount:
                return [b"eos"]
            return [b"]"] + [b"[" + el + b"]" for el in
                self._create_form_prefixes(context)] + \
                [b"[" + el + b"[" for el in
                self._create_relation_prefixes(context)]
        elif context.last_unused_tokens.total() == 0:
            return [b"]"]
        elif context.generated_bytes.endswith(b"["):
            return [el + b"]" for el in
                self._create_form_prefixes(context)] + \
                [el + b"[" for el in
                self._create_relation_prefixes(context)] 
        else:
            last_el_text = context.generated_bytes.split(b"[")[-1]
            s_prefixes = [el[len(last_el_text):] + b"]" for el in
                self._create_form_prefixes(context)
                if el.startswith(last_el_text)] + \
                [el[len(last_el_text):] + b"[" for el in
                self._create_relation_prefixes(context)
                if el.startswith(last_el_text)]
            if b"[" in s_prefixes: # TODO: укорить
                s_prefixes.remove(b"[")
                s_prefixes += [b"[" + el + b"]" for el in
                    self._create_form_prefixes(context)] + \
                        [b"[" + el + b"[" for el in
                        self._create_relation_prefixes(context)] 
            # TODO: if "]" in s_prefixes: - ввести "]" + конец ?
            #    s_prefixes.remove("]")
            return s_prefixes


from genlm.backend.tokenization import decode_vocab
import marisa_trie

class PrefixFinder:
    def __init__(self, bytes_to_id):
        super().__init__()
        self.bytes_to_id = bytes_to_id
        self.trie = marisa_trie.BinaryTrie(list(bytes_to_id.keys()))

    def __call__(self, prefixes: str):
        allow_ids = []
        print(f"{prefixes=}")
        for target in prefixes:
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
    def __init__(self, bytes_to_id, eos_ids, relations):
        self.prefix_generator = PrefixGenerator(relations)
        self.prefix_checker = PrefixFinder(bytes_to_id)
        self.eos_ids = eos_ids

    def check(self, context):
        return True

    def __call__(self, logits, context):
        prefixes = self.prefix_generator(context)
        if prefixes == [b"eos"]:
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
    
    def __init__(self, id_to_bytes, applying_max_amount):
        self.id_to_bytes = id_to_bytes
        self.applying_max_amount = applying_max_amount
        
    def check(self, context):
        if not self.applying_max_amount:
            return False
        return context.check_all_open() and \
            not context.check_all_end() and \
            len(context.generated_bytes) > 0 and \
            context.generated_bytes[-1] == END_BYTE
        # Generate some last "]"

    def __call__(self, logits, context):
        bracket_diff = context.op_amount - context.end_amount
        print("Forcing last ]")
        codes_with_logits = [(token_bytes, token_id, float(logits[token_id]))
                             for token_id, token_bytes in self.id_to_bytes.items()]
        logits[:] = -torch.inf
        for token_bytes, token_id, token_logit in codes_with_logits:
            if set(token_bytes) == {b"]"} and bracket_diff - len(token_bytes) >= 0:
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
    def __init__(self, id_to_bytes):
        self.id_to_bytes = id_to_bytes

    def check(self, context):
        return len(context.generated_bytes) > 0 and \
            context.generated_bytes[-1] == OPEN_BYTE
        # After "[" any bracket subtoken is restricted
        
    def __call__(self, logits, context):
        for token_id, token_bytes in self.id_to_bytes.items():
            if token_bytes[0] in {b"[", b"]"}:
                logits[token_id] = float('-inf')
                #print(token_id, token_text)
        #print("Restriction for bracket after open bracket")
        return logits
        
class RestrictTextAfterEndConstraint(Constraint):
    def __init__(self, id_to_bytes, eos_ids):
        self.id_to_bytes = id_to_bytes
        self.eos_ids = eos_ids

    def check(self, context):
        return len(context.generated_bytes) > 0 and \
            context.generated_bytes[-1] == END_BYTE
        
    def __call__(self, logits, context): # TODO: запрет нужен для всех!!!
        mask_tensor = torch.full(logits.shape, -float('inf'), device=logits.device)
        for token_id, token_bytes in self.id_to_bytes.items():
            if token_bytes[0] == END_BYTE or token_bytes[0] == OPEN_BYTE:
                mask_tensor[token_id] = 0
        for token_id in self.eos_ids:
            mask_tensor[token_id] = 0
        logits += mask_tensor
        return logits

        
class RestrictBalanceBracketConstraint(Constraint):
    """"""
    
    def __init__(self, id_to_bytes, applying_max_amount, soft_max_amount):
        self.id_to_bytes = id_to_bytes
        self.applying_max_amount = applying_max_amount
        self.soft_max_amount = soft_max_amount
        
    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        bracket_diff = context.op_amount - context.end_amount
        for token_id, token_bytes in self.id_to_bytes.items():
            inf_flag = False
            for i in range(1, len(token_bytes)):
                token_slice = token_bytes[:i]
                if bracket_diff + token_slice.count(b"[") - token_slice.count(b"]") < 1:
                    # Ошибка вида "[Остается][" при добавлении "]["
                    inf_flag = True
            if not inf_flag:
                new_bracket_diff = bracket_diff + \
                    token_bytes.count(b"[") - token_bytes.count(b"]")
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
    
    def __init__(self, id_to_bytes, applying_max_amount):
        self.id_to_bytes = id_to_bytes
        self.applying_max_amount = applying_max_amount
        
    def check(self, context):
        if not self.applying_max_amount:
            return False
        return context.check_all_open() and not context.check_all_end()
    
    def __call__(self, logits, context):
        print("Restriction for [")
        for token_id, token_bytes in self.id_to_bytes.items():
            if b"[" in token_bytes:
              logits[token_id] = float('-inf')
              #print(f"Restricted {token_id} ({token_text})")
        return logits

class RestrictErrorTokenConstraint(Constraint):
    """"""
    
    def __init__(self, id_to_bytes):
        self.error_indexes = []
        for token_id, token_bytes in id_to_bytes.items():
            error_token_flag = False
            for t_i, t in enumerate(token_bytes):
               if t == END_BYTE:
                   if t_i != len(token_bytes) - 1 and \
                     token_bytes[t_i + 1] not in {b"]", b"["}:
                       error_token_flag = True
                       break
               elif t == OPEN_BYTE:
                   if t_i != len(token_bytes) - 1 and \
                     token_bytes[t_i + 1] == END_BYTE: # "[]"
                       error_token_flag = True
                       break
                   if t_i != 0 and token_bytes[t_i - 1] != END_BYTE:
                       error_token_flag = True
                       break
               #elif not (t.isalnum() or t in string.punctuation or
               #  t in " “”„«»—–…№" or 0xFF00 <= ord(t) < 0xFFEF):
               #    error_token_flag = True
               #    break
                
            if error_token_flag:
                self.error_indexes.append(token_id)
        for token_bytes in [b"<think>", b"</think>", b"<|im_start|>"]:
            if token_bytes in id_to_bytes:
                self.error_indexes.append(id_to_bytes[token_bytes])

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
    return b"[" not in x[1:-1] and b"]" not in x[1:-1] and \
        b"|T" not in x and b"|C" not in x and b"|W" not in x and b"|E" not in x

def change_text(change_fun, right_border_symbol, s):
    #print(f"{right_border_symbol=}")
    ready_border = 0
    left_border_in_fragment = s[ready_border:].find(b"[")
    right_border_in_fragment = s[ready_border + left_border_in_fragment + 1:].find(right_border_symbol)
    while left_border_in_fragment != -1 and right_border_in_fragment != -1:
        left_border = ready_border + left_border_in_fragment
        change_text = change_fun(s[left_border:left_border + 1 + right_border_in_fragment + 1])
        #print(f"{left_border=} {right_border_in_fragment=} {s[left_border:left_border + 1 + right_border_in_fragment + 1]} {change_text}")
        new_s = s[:left_border] + change_text + s[left_border + 1 + right_border_in_fragment + 1:]
        if b"|E" in change_text:
            return new_s
        ready_border = left_border + 1 # Все, что левее left_border - неизменно
        s = new_s
        #print(f"start with ready_border: {s[ready_border:]} {s[ready_border:].find(b"[")}")
        left_border_in_fragment = s[ready_border:].find(b"[")
        if left_border_in_fragment == -1:
            break
        #print(f"{left_border_in_fragment=}")
        #print(f"{s[ready_border + left_border_in_fragment + 1:]} {s[ready_border + left_border_in_fragment + 1:].find(right_border_symbol)}")
        right_border_in_fragment = s[ready_border + left_border_in_fragment + 1:].find(right_border_symbol)
        #print(f"{right_border_in_fragment=}")
    return s

def last_tokens_is_form_prefix(last_tokens, unused_tokens):
    if unused_tokens is None:
        return False
    if len([1 for tok in unused_tokens if tok.startswith(last_tokens)]) != 0:
        return True
    return False

def last_tokens_is_relation_prefix(last_tokens, allow_relations):
    return len([rel for rel in allow_relations if rel.startswith(last_tokens)]) != 0

def fold_bracket_seq(re_before_right_open: bytes, bytes_after_right_open:bytes=b"",
                     allow_relations=None, unused_tokens_param=None, drop_tokens=False):
    # re_before_right_open - до крайней правой ], включительно (можем сворачивать, все байты уже получены)
    # bytes_after_right_open - после крайней правой ], еще могут прийти байты, нельзя переводить в символы
    #print(f"{re_before_right_open=}")
    #print(f"{bytes_after_right_open=}")
    #print(f"{allow_relations=}")
    #print(f"{unused_tokens_param=}")
    #print(f"{drop_tokens=}")
    # >>> fold_bracket_seq("[roet[")
    # '|E'
    if drop_tokens:
        unused_tokens = unused_tokens_param
    else:
        unused_tokens = copy.deepcopy(unused_tokens_param)

    if unused_tokens is not None:
        initial_unused_amount = unused_tokens.total()

    re_before_right_open = re_before_right_open.replace(b" ", b"")

   
    if re_before_right_open[:6] == b"[root[":
        re_before_right_open = b"[|T[" + re_before_right_open[6:]
    elif not re_before_right_open.startswith(b"[|T"):
        if len(re_before_right_open) < 6:
            if b"[root[".startswith(re_before_right_open + bytes_after_right_open): # bytes_after_right_open - байты, проверка на уровне байтов # TODO: проверить логику
                return re_before_right_open
            return b"|E"
        else:
            return b"|E"

    # >>> fold_bracket_seq('[root[К', {'nmod'}, Counter(['Дым', 'Дом']))
    # '[|T[|E'
    # >>> fold_bracket_seq('[root[Дом]]', {'nmod'}, Counter(['Дым', 'Дом'])) - |
    #TODO1
    if not bytes_after_right_open.startswith(b"|T") and \
        b"]" not in bytes_after_right_open and unused_tokens is not None and \
            (not last_tokens_is_form_prefix(bytes_after_right_open, unused_tokens) and
            not last_tokens_is_relation_prefix(bytes_after_right_open, allow_relations)):
        return re_before_right_open + b"|E"
    last_re_level_ind = re_before_right_open.rfind(b"[|T")
    if last_re_level_ind == -1:
        last_re_level_ind = 0
    last_re_level = re_before_right_open[last_re_level_ind:]
    if last_re_level.count(b"|W") > 1:
        return re_before_right_open + b"|E"
    if last_re_level.count(b"|W") == 1 and \
        (last_tokens_is_form_prefix(bytes_after_right_open, unused_tokens) and
            not last_tokens_is_relation_prefix(bytes_after_right_open, allow_relations)):
        # [|T|W[|T|W|C|C|C[$ - E (два на уровне будут)
        return re_before_right_open + b"|E"
    if allow_relations is not None:
        f_rel = lambda x: (b"[|T[" if x[1:-1] in allow_relations else b"[|E[") if constant_check(x) else x

    else:
        f_rel = lambda x: b"[|T[" if constant_check(x) else x

    if unused_tokens is not None:
        def f_form_drop(x):
            if not constant_check(x):
                return x
            form_text = x[1:-1]
            if form_text in unused_tokens and unused_tokens[form_text] > 0:
                unused_tokens[form_text] -= 1
                if unused_tokens[form_text] <= 0:
                    unused_tokens.pop(form_text)
                return b"[|T]"
            else:
                return b"[|E]"
        f_form = f_form_drop
    else:
        f_form = lambda x: b"[|T]" if constant_check(x) else x

    # >>> fold_bracket_seq('[root[Дом][nmod[Тен]]]', {'nmod'}, Counter(['Дым', 'Дом', 'Тен'])) - |C
    re_before_right_open = change_text(f_form, b"]", re_before_right_open)
    #print(f"after change_text 1: {re_before_right_open=}")
    if b"|E" in re_before_right_open:
        return re_before_right_open
    re_before_right_open = change_text(f_rel, b"[", re_before_right_open)
    #print(f"after change_text 2: {re_before_right_open=}")
    if b"|E" in re_before_right_open:
        return re_before_right_open

    f_check_level_two_words = lambda x: b"|E" if x.count(b"|W") > 1 else x

    # Против [|T|C|W|W[
    # Только для закрытых уровней. Открытый рассматривается раньше
    re_before_right_open = change_text(f_check_level_two_words, b"[", re_before_right_open)
    #print(f"after change_text 3: {re_before_right_open=}")

    if b"|E" in re_before_right_open:
        return re_before_right_open

    if unused_tokens is not None:
        unused_amount_after_fold = initial_unused_amount - re_before_right_open.count(b"[|T]") # обещанный приход

    re_before_right_open = re.sub(rb"\[\|T\]", b"|W", re_before_right_open)
    #print(f"after re sub: {re_before_right_open=}")


    while b"|E" not in re_before_right_open and b"]" in re_before_right_open and re_before_right_open[0] == OPEN_BYTE:
        #print(f"while: {re_before_right_open=}")
        end_bracket = re_before_right_open.find(b"]")
        op_bracket = re_before_right_open[:end_bracket].rfind(b"[")
        if re_before_right_open[op_bracket + 1:op_bracket + 3] != b"|T" or re_before_right_open[op_bracket + 2: end_bracket].count(b"|W") != 1:
            re_before_right_open = re_before_right_open[:op_bracket + 1] + b"|E" + re_before_right_open[end_bracket:]
            break
        else:
            re_before_right_open = re_before_right_open[:op_bracket] + b"|C" + re_before_right_open[end_bracket + 1:]
    #print(f"after while: {re_before_right_open=}")

    if re_before_right_open != b"" and re_before_right_open[0] != OPEN_BYTE and re_before_right_open != b"|C":
        re_before_right_open = b"|E"

    #print(f"after !=: {re_before_right_open=}")
    if unused_tokens is not None:
        # [|T|W[|T|C|W|C[|T[Home]] - Counter({'Home': 1})
        # after text [|T|W[|T|C|W|C[|T[|T]]
        # unused_amount_after_fold=0 level_openers=3 level_closers=3
        level_openers = re_before_right_open.replace(b"[|T]", b"").count(b"[|T") # расход
        level_closers = re_before_right_open.count(b"|W") + re_before_right_open.count(b"[|T]") # уже случившийся приход
        #print(f"{unused_amount_after_fold=} {level_openers=} {level_closers=}")
        if level_openers > unused_amount_after_fold + level_closers:
            return b"|E" # расход привысил потенциальный и реальный приход
        if level_openers == unused_amount_after_fold + level_closers:
            last_level_start = re_before_right_open.rfind(b"[|T")
            if last_level_start != -1:
                last_level_text = re_before_right_open[last_level_start:]
                #print(f"{last_level_text=}")
                if level_openers == level_closers:
                # Бюджет сведен
                    if last_level_text.count(b"[") > 1:
                        return b"|E"
            # >>> fold_bracket_seq("[|T|W|C[|T|C|C|W|C[|T|C|W[|T|W", ENGLISH_RELATIONS, Counter(), True)
            # '[|T|W|C[|T|C|C|W|C[|T|C|W[|T|W'
                else:   
            # Бюджет еще не сведен
            # Есть шансы свести бюджет, но нужно действовать аккуратно
            # Нельзя открывать новые уровни, только новые слова
                    # На последнем уровне уже есть |W, второе |W не может быть,
                    # все [ - открывают уровень
                    if b"|W" in last_level_text:
                        if last_level_text.count(b"[") > 1:
                            return b"|E"
                    else: # Нужно сгенерировать слово.
                        # За запрет генерации типа связи отвечает генератор префиксов
                        # Но на всякий случай нужно проверить
                        if re_before_right_open[-2:] != b"|C" and re_before_right_open[-2:] != b"|T":
                            # [|T|W|C[|T|C|C, Counter([.]) - ok
                            if not last_tokens_is_form_prefix(bytes_after_right_open, unused_tokens): # Не выносить, здесь новые unused_tokens
                                return b"|E"
    #print(f"after budget: {re_before_right_open=}")

    #>>> fold_bracket_seq("[|T|C|C|W|C[|T|C|C[.]][", ENGLISH_RELATIONS, Counter({'.': 1}), True)
    # after text [|T|C|C|W|C[|T|C|C[|T]][
    # after fold [|T|C|C|W|C|C[
    # unused_amount_after_fold=0 level_openers=1 level_closers=1
    #'|E'
    # >>> fold_bracket_seq("[|T|C|C|W|C[|T|C|C[.]][", ENGLISH_RELATIONS, Counter({'.': 1}), False)
    # after text [|T|C|C|W|C[|T|C|C[|T]][
    # after fold [|T|C|C|W|C|C[
    # unused_amount_after_fold=0 level_openers=1 level_closers=1
    #'|E'
    # >>> fold_bracket_seq("[|T|C|W[|T|C|W|C[|T[.]", ENGLISH_RELATIONS, Counter({'.': 1}), True)
    # after text [|T|C|W[|T|C|W|C[|T[|T]
    # unused_amount_after_fold=0 level_openers=1 level_closers=3
    # '[|T|C|W[|T|C|W|C[|T|W'
    # >>> fold_bracket_seq("[|T|C|W[|T|C|W|C[|T[.]", ENGLISH_RELATIONS, Counter({'.': 1}), False)
    # after text [|T|C|W[|T|C|W|C[|T[|T]
    # unused_amount_after_fold=0 level_openers=1 level_closers=3
    # '[|T|C|W[|T|C|W|C[|T|W'
                    
    # >>> fold_bracket_seq("[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[fir", ENGLISH_RELATIONS, Counter({'of': 1, 'my': 1, 'firm': 1}), True)
    # after text [|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[fir
    # '[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[fir'
    # >>> fold_bracket_seq("[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[firm", ENGLISH_RELATIONS, Counter({'of': 1, 'my': 1, 'firm': 1}), True)
    # after text [|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[firm
    # '[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[firm'
    #>>> fold_bracket_seq("[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[frm", ENGLISH_RELATIONS, Counter({'of': 1, 'my': 1, 'firm': 1}), True)
    # '[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[|E'
    #>>> fold_bracket_seq("[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[frm", ENGLISH_RELATIONS, Counter({'of': 1, 'my': 1, 'firm': 1}), False)
    # '[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[|E'
    # >>> fold_bracket_seq("[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[firm", ENGLISH_RELATIONS, Counter({'of': 1, 'my': 1, 'firm': 1}), False)
    # after text [|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[firm
    # '[|T|C[|T|C|C|W[|T|C|C|C|W[|T[|T[firm'

    # >>> fold_bracket_seq("[|T[|T|W|C|C|C[|T|C|C[of][", ENGLISH_RELATIONS, Counter([',', 'of']))
    #'|E'
    #>>> fold_bracket_seq("[|T[|T|W|C|C|C[|T|C|C[of][", ENGLISH_RELATIONS, Counter([',', 'of']), True)
    #'|E'

    return re_before_right_open


class RestrictUncorrectLevelConstraint(Constraint):

    def __init__(self, id_to_bytes, relations):
        self.relations = relations
        self.id_to_bytes = id_to_bytes

    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        #print("Restrictions for grct levels")
        #for token_text, token_id in self.partial_bracket_codes: # TODO: Только для оставшихся разрешенными
            # TODO: Куда ставить проверку после проверки по префиксам ?

        valid_indexes = []
        iter_i = 0
        max_logits = None
        MIN_VALUE_LEN = 3 # TODO: больше него нельзя делать выбор элементов
        ITER_SIZE = 10
        while len(valid_indexes) < MIN_VALUE_LEN and (max_logits is None or not max_logits[-1].isinf()):
            max_logits, max_indices = torch.topk(logits, k = ITER_SIZE * (iter_i + 1))
            #print(f"{max_logits=}")
            #print(f"{max_indices=}")
            #print(f"{iter_i=}")
            for token_i, token_id in enumerate(max_indices[ITER_SIZE * iter_i:]):
                if max_logits[token_i + ITER_SIZE * iter_i].isinf():
                    break

                re_s, bytes_after_right_open =  create_bytes_after_right_open(context.re_before_right_open,
                    context.bytes_after_right_open, self.id_to_bytes[token_id.item()])

                fold_res = fold_bracket_seq(re_s, bytes_after_right_open,
                            self.relations, context.last_unused_tokens, False)
                #print(f"{re_s=}\n{bytes_after_right_open=}\n{fold_res=}")
                if b"|E" not in fold_res:
                        # TODO: Проверка с учетом типа связи/формы и без них
                        # Проверка, без изменения набора токенов
                    valid_indexes.append(token_id.item())
            iter_i += 1
            #print(f"{valid_indexes=}")
        valid_mask = torch.zeros_like(logits, dtype=torch.bool)
        valid_mask[valid_indexes] = True
        logits[~valid_mask] = float('-inf')
        return logits

# '[|T[|T[|T[' + 'стол]' тоже нужно соединять и сворачивать, а right_open_bracket_pos будет пустым
def create_bytes_after_right_open(re_before_right_open, old_bytes_after_right_open, last_token_bytes):
    right_open_bracket_pos = max(last_token_bytes.rfind(b"["), last_token_bytes.rfind(b"]"))
    # TODO: check and rename
    if right_open_bracket_pos != -1:
        before_open_bracket = last_token_bytes[:right_open_bracket_pos + 1]
        after_open_bracket = last_token_bytes[right_open_bracket_pos + 1:]
    else:
        before_open_bracket, after_open_bracket = b"", last_token_bytes

    if before_open_bracket:
        re_s = re_before_right_open + \
                old_bytes_after_right_open + before_open_bracket
        bytes_after_right_open = after_open_bracket
    else:
        re_s = re_before_right_open
        bytes_after_right_open = old_bytes_after_right_open + after_open_bracket
    return re_s, bytes_after_right_open

class GenerationContext:
    def __init__(self, max_op_bracket, relations, last_unused_tokens):
        self.generated_bytes = b""
        self.op_amount = 0
        self.end_amount = 0
        self.max_op_bracket = max_op_bracket
        self.last_unused_tokens = Counter([v.encode("utf-8") for v in last_unused_tokens])
        self.bytes_after_right_open = b""
        self.re_before_right_open = b""
        self.relations = relations

    def update_context(self, last_token_bytes):
        self.generated_bytes += last_token_bytes

        self.op_amount += last_token_bytes.count(b"[")
        self.end_amount += last_token_bytes.count(b"]")
        
        re_s, self.bytes_after_right_open =  create_bytes_after_right_open(self.re_before_right_open,
                            self.bytes_after_right_open, last_token_bytes)

        self.re_before_right_open = fold_bracket_seq(re_s, self.bytes_after_right_open, # TODO: не будет работать для строк с |, в SynTagRus нет
            self.relations, self.last_unused_tokens, True)

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
        relation_language = logit_params.get("relations")
        self.relations = [r.encode("utf-8") for r in RELATION_DICT[relation_language]]
        self.max_op_bracket = None
        self.tokenizer = tokenizer

        applying_first_root = "root" in optional_constraints
        #print(f"applying_first_root: {applying_first_root}")
        applying_max_amount = "max_amount" in optional_constraints
        #print(f"applying_max_amount: {applying_max_amount}") # Ровно заданное количество [
        soft_max_amount = "soft_max_amount" in optional_constraints
        #print(f"soft_max_amount: {soft_max_amount}") # Не более заданного количества [
        if soft_max_amount:
            applying_max_amount = True

        byte_vocab, _ = decode_vocab(tokenizer)
        bytes_to_id = {subtoken.byte_string: subtoken.token_id
                                    for subtoken in byte_vocab}
        self.id_to_bytes = {subtoken.token_id: subtoken.byte_string
                                    for subtoken in byte_vocab}

        self.force_finish_constraints = ForceFinishConstraint(tokenizer.eos_token_id,
            applying_max_amount, soft_max_amount)
        self.force_end_constraints = ForceEndConstraint(self.id_to_bytes, applying_max_amount)

        eos_ids = [tokenizer.old_eos_token_id, tokenizer.eos_token_id]
        print(f"eos_ids: {eos_ids}")
        self.prefix_constraints = PrefixConstraint(bytes_to_id, eos_ids, self.relations)

        self.restrict_bracket_after_open_constraints = RestrictBracketAfterOpenConstraint(self.id_to_bytes)
        self.restrict_text_after_end_constraints = RestrictTextAfterEndConstraint(self.id_to_bytes, eos_ids)
        self.restrict_balance_constraints = RestrictBalanceBracketConstraint(self.id_to_bytes,
            applying_max_amount, soft_max_amount)
        self.restrict_open_constraints = RestrictOpenConstraint(self.id_to_bytes, applying_max_amount)
        self.restrict_error_constraints = RestrictErrorTokenConstraint(self.id_to_bytes)
        self.restrict_unbalanced_eos_constraints = RestrictUnbalancedEOSConstraint(eos_ids)

        self.restrict_uncorrect_level_constraints = RestrictUncorrectLevelConstraint(
            self.id_to_bytes, self.relations)

        self.mul_coeff = logit_params.get("mul_coeff", 1)
        self.add_coeff = logit_params.get("add_coeff", 0)
        self.prev_context = None


    def create_new_context(self, input_tokens): 
        max_op_bracket = (self.mul_coeff * len(input_tokens) + \
            self.add_coeff) * 2
        self.prev_context =  GenerationContext(max_op_bracket,
                    self.relations, input_tokens)

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
        if len(token_ids) > 0:
            last_token_bytes = self.id_to_bytes[token_ids[-1]]
        else:
            last_token_bytes = b""

        #torch.cuda.synchronize()

        #tf = time.perf_counter()
        #time_list.append(("decode", tf - ts))

        #torch.cuda.synchronize()
        #ts = time.perf_counter()
        self.prev_context.update_context(last_token_bytes)
        context = self.prev_context
        print(f"{self.prev_context.__dict__=}")

        # max_op_bracket в контекст, т.к. используется в ForceClosingConstraint,
        # а его нельзя создавать до create_new_context

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
