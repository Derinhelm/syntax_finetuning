from collections import Counter


import torch

from src.inference_functions.fold_seq import fold_bracket_seq, RestrictUncorrectLevelConstraint, \
    create_bytes_after_right_open, OPEN_BYTE, END_BYTE, RELATION_DICT
from src.inference_functions.constraint import Constraint

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
        #print("Forcing last ]")
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
        #print("Restriction for [")
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

    def update_context(self, last_token_bytes):
        self.generated_bytes += last_token_bytes

        self.op_amount += last_token_bytes.count(b"[")
        self.end_amount += last_token_bytes.count(b"]")
        
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


    def create_new_context(self, input_tokens): 
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
        max_logits, _ = torch.topk(logits, k = 1) # Нет разрешенных субтокенов
        if max_logits[0].isinf():
            logits = self.force_finish_constraints(logits, context)

        return logits
