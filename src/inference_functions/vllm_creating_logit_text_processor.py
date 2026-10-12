
from collections import Counter
from cachetools import LRUCache

import itertools

import torch

from inference_functions.fold_seq import fold_bracket_seq, RestrictUncorrectLevelConstraint, \
    create_bytes_after_right_open, OPEN_BYTE, END_BYTE, RELATION_DICT
from inference_functions.constraint import Constraint

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
            if END_BYTE not in context.bytes_after_right_open and context.bytes_after_right_open in self.relations:
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
        self._cache = LRUCache(maxsize=256)

    def __call__(self, prefixes: list[bytes]):
        allow_ids = set()
        for target in prefixes:
            if target in self._cache:
                allow_ids.update(self._cache[target])
            else:
                # Условие 1: токены, которые являются префиксом target
                new_ids = { self.bytes_to_id[b] for b in \
                    itertools.chain(self.trie.iter_prefixes(target), self.trie.iterkeys(target)) }
                allow_ids.update(new_ids)
                #print(f"{allow_ids=}")
                self._cache[target] = new_ids

        return list(allow_ids)

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
        self.applying_max_amount = applying_max_amount
        self.end_tokens = []
        for token_id, token_bytes in id_to_bytes.items():
            if len(token_bytes) > 0 and set(token_bytes) == {END_BYTE}:
                self.end_tokens.append((token_id, len(token_bytes))) # Токенов мало, поэтому не словарь
        print(f"{self.end_tokens=}")
        
    def check(self, context):
        if not self.applying_max_amount:
            return False
        return  len(context.generated_bytes) > 0 and \
                context.generated_bytes[-1] == END_BYTE and \
                context.check_all_open() and \
                not context.check_all_end()
            
        # Generate some last "]"
    def __call__(self, logits, context):
        bracket_diff = context.op_amount - context.end_amount
        valid_ids = [
            token_id for token_id, length in self.end_tokens 
            if length <= bracket_diff
        ]

        new_logits = torch.full_like(logits, -torch.inf)
        new_logits[valid_ids] = logits[valid_ids]

        return new_logits

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
        wrong_ids = [token_id for token_id, token_bytes in id_to_bytes.items()
                          if token_bytes[0] in {OPEN_BYTE, END_BYTE}]
        self.wrong_ids = torch.tensor(wrong_ids, dtype=torch.long)

    def check(self, context):
        return len(context.generated_bytes) > 0 and \
            context.generated_bytes[-1] == OPEN_BYTE
        # After "[" any bracket subtoken is restricted
        
    def __call__(self, logits, context):
        logits[self.wrong_ids] = float('-inf')
                #print(token_id, token_text)
        #print("Restriction for bracket after open bracket")
        return logits
        
class RestrictTextAfterEndConstraint(Constraint):
    def __init__(self, id_to_bytes, eos_ids):
        self.allowed_ids = [ token_id for token_id, token_bytes in id_to_bytes.items()
                           if token_bytes[0] == END_BYTE or token_bytes[0] == OPEN_BYTE ] + eos_ids

    def check(self, context):
        return len(context.generated_bytes) > 0 and \
            context.generated_bytes[-1] == END_BYTE
        
    def __call__(self, logits, context): # TODO: запрет нужен для всех!!!
        allowed_logits = logits[self.allowed_ids].clone()
        logits.fill_(-float('inf'))
        logits[self.allowed_ids] = allowed_logits
        return logits

        
class RestrictBalanceBracketConstraint(Constraint):
    """"""
    
    def __init__(self, id_to_bytes, applying_max_amount, soft_max_amount):
        self.id_to_bytes = id_to_bytes
        self.applying_max_amount = applying_max_amount
        self.soft_max_amount = soft_max_amount
        full_deltas = []
        bracketed_token_ids = []
        min_deltas = []
        for token_id, token_bytes in self.id_to_bytes.items():
            if OPEN_BYTE in token_bytes or END_BYTE in token_bytes:
                min_delta = None
                for i in range(1, len(token_bytes)):
                    token_slice = token_bytes[:i]
                    delta = token_slice.count(OPEN_BYTE) - token_slice.count(END_BYTE)
                    if min_delta is None or delta < min_delta:
                        min_delta = delta
                bracketed_token_ids.append(token_id)
                min_deltas.append(min_delta)
                full_deltas.append(token_bytes.count(OPEN_BYTE) - token_bytes.count(END_BYTE))
        self.bracketed_token_ids = torch.tensor(bracketed_token_ids, dtype=torch.long)
        self.full_deltas = torch.tensor(full_deltas, dtype=torch.int32)
        self.min_deltas = torch.tensor(min_deltas, dtype=torch.int32)


    def check(self, context):
        return True
    
    def __call__(self, logits, context):
        bracket_diff = context.op_amount - context.end_amount
        if self.applying_max_amount and not self.soft_max_amount: # Есть жесткое ограничение по количеству открытых скобок
            if not context.check_all_open(): # И не все [ сгенерированы
                # То есть нельзя делать diff = 0
                threshold = 1
            else: # Все [ сгенерированы, можно diff = 0
                threshold = 0
        else: 
            # Ограничений по количеству скобок нет, можно уходить в ноль (тогда потом будет force end)
            threshold = 0

        # Ошибка вида "[Остается][" при добавлении "]["
        min_filter = (bracket_diff + self.min_deltas < 1)
        full_filter = (bracket_diff + self.full_deltas < threshold)
        wrong_ids = self.bracketed_token_ids[min_filter | full_filter]
        logits[wrong_ids] = float('-inf')
        return logits

class RestrictOpenConstraint(Constraint):
    """"""
    
    def __init__(self, id_to_bytes, applying_max_amount):
        self.applying_max_amount = applying_max_amount
        self.wrong_ids = [ token_id for token_id, token_bytes in id_to_bytes.items()
                           if OPEN_BYTE in token_bytes ] 

        
    def check(self, context):
        if not self.applying_max_amount:
            return False
        return context.check_all_open() and not context.check_all_end()
    
    def __call__(self, logits, context):
        logits[self.wrong_ids] = float('-inf')
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
                     token_bytes[t_i + 1] not in {END_BYTE, OPEN_BYTE}:
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
        #print("Restriction for eos (because of unbalancing)")
        logits[self.eos_ids] = -torch.inf
        return logits

class GenerationContext:
    def __init__(self, max_op_bracket, relations, last_unused_tokens, skip_text):
        self.generated_bytes = b""
        self.op_amount = 0
        self.end_amount = 0
        self.max_op_bracket = max_op_bracket
        self.last_unused_tokens = Counter([v.encode("utf-8") for v in last_unused_tokens])
        self.bytes_after_right_open = b""
        self.re_before_right_open = b""
        self.relations = relations
        self.skip_text = skip_text

    def snapshot(self):
        return  { 'generated_bytes': self.generated_bytes
                , 'op_amount': self.op_amount
                , 'end_amount': self.end_amount
                , 'max_op_bracket': self.max_op_bracket
                , 'last_unused_tokens': dict(self.last_unused_tokens)
                , 'bytes_after_right_open': self.bytes_after_right_open
                , 're_before_right_open': self.re_before_right_open
                , 'relations': self.relations
                , 'skip_text': self.skip_text
                }

    def update_context(self, last_token_bytes):
        self.generated_bytes += last_token_bytes

        self.op_amount += last_token_bytes.count(OPEN_BYTE)
        self.end_amount += last_token_bytes.count(END_BYTE)
        
        re_s, self.bytes_after_right_open =  create_bytes_after_right_open(self.re_before_right_open,
                            self.bytes_after_right_open, last_token_bytes)

        if not self.skip_text:
            self.re_before_right_open = fold_bracket_seq(re_s, self.bytes_after_right_open, # TODO: не будет работать для строк с |, в SynTagRus нет
                self.relations, self.last_unused_tokens, True)
        else:
            self.re_before_right_open = fold_bracket_seq(re_s, self.bytes_after_right_open)

    def check_all_open(self):
        return self.op_amount == self.max_op_bracket
    
    def check_all_end(self):
        return self.end_amount == self.max_op_bracket


class TextBracketLogitsProcessor:
    def __init__(self, tokenizer, logit_params):
        optional_constraints = logit_params.get("optional_constraints", set())
        self.skip_text = logit_params.get("skip_text", False)
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
            self.id_to_bytes, self.relations, self.skip_text)

        self.mul_coeff = logit_params.get("mul_coeff", 1)
        self.add_coeff = logit_params.get("add_coeff", 0)
        self.prev_context = None
        self.current_contexts = []


    def create_new_context(self, input_tokens):
        if self.current_contexts:
            print(*self.current_contexts, sep='\n')
            self.current_contexts = []
        max_op_bracket = (self.mul_coeff * len(input_tokens) + \
            self.add_coeff) * 2
        self.prev_context =  GenerationContext(max_op_bracket,
                    self.relations, input_tokens, self.skip_text)

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
        self.current_contexts.append(context.snapshot())

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
            if not self.skip_text and self.prefix_constraints.check(context):
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
        if torch.isneginf(logits.max()):
            logits = self.force_finish_constraints(logits, context)
        return logits
