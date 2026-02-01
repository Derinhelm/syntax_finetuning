import copy

def shift_token_id(tokens_param):
    # TODO: Не работает, если id - "1.3"
    tokens = copy.deepcopy(tokens_param)
    first_token_shift = 0
    for i, t in enumerate(tokens):
        if t["id"] == '1':
            first_token_shift = i
            shift_id = str(int(t["id"]) + first_token_shift)
        if t["parent_id"] != '0':
            shift_parent_id = str(int(t["parent_id"]) + first_token_shift)
        else:
            shift_parent_id = '0'
            t["id"] = shift_id
            t["parent_id"] = shift_parent_id
    return tokens

    
from functions import create_sent_be_nodes, create_sent_be_edges # TODO
from collections import OrderedDict
import pandas as pd

def calculate_stat(be_gold, be_res_p):
    extra_gold = len(be_gold.keys() - be_res_p.keys())
    extra_parser = len(be_res_p.keys() - be_gold.keys())
    tokenisation_matched = len({t_be for t_be in be_res_p.keys() & be_gold.keys()})
    parent_matched = len({t_be for t_be in be_res_p.keys() & be_gold.keys()
      if be_res_p[t_be][0] == be_gold[t_be][0]})
    relation_matched = len({t_be for t_be in be_res_p.keys() & be_gold.keys()
      if be_res_p[t_be] == be_gold[t_be]})
      
    # TODO: Заменять null на 0
    align_uas_precision = parent_matched / (extra_parser + tokenisation_matched)
    align_uas_recall = parent_matched / (extra_gold + tokenisation_matched)
    align_las_precision = relation_matched / (extra_parser + tokenisation_matched)
    align_las_recall = relation_matched / (extra_gold + tokenisation_matched)
    align_uas_f = (2 * align_uas_precision * align_uas_recall) / (align_uas_precision + align_uas_recall)
    align_las_f = (2 * align_las_precision * align_las_recall) / (align_las_precision + align_las_recall)
    return align_uas_f, align_las_f

def calculate(gold_text, gold_tokens_param, parser_tokens_params):
    # TODO: Только, если текст не изменялся
    gold_tokens = shift_token_id(gold_tokens_param)
    print(gold_tokens)
    be_sent, be_token_dict_gold = create_sent_be_nodes(gold_text, gold_tokens,
        lambda text: text.lower())
    be_gold = create_sent_be_edges(be_sent)

    parser_tokens = shift_token_id(parser_tokens_params)
    # TODO: For DeepPavlov
    # transform_fun = lambda text: text.lower().replace('``', '"').replace("''", '"')
    transform_fun = lambda text: text.lower().replace("''", '"')
    be_sent, _ = create_sent_be_nodes(gold_text, parser_tokens,
        transform_fun)
    be_parser = create_sent_be_edges(be_sent)
    uas, las = calculate_stat(be_gold, be_parser)
    return uas, las
