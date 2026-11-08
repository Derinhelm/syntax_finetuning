from abc import ABC, abstractmethod
import copy
import re
import string
from typing import List, Tuple
from collections import Counter


import torch


from inference_functions.constraint import Constraint

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

    def __init__(self, id_to_bytes, relations, skip_text):
        self.relations = relations
        self.id_to_bytes = id_to_bytes
        self.skip_text = skip_text

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

                if self.skip_text:
                    fold_res = fold_bracket_seq(re_s, bytes_after_right_open)
                else:
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
