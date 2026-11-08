from collections import Counter
from copy import deepcopy

def process_trees(gold_text, gold_tree, pred_tree, metric_type):
    if metric_type == "difference_easy":
        from metric_functions.edge_creating.creating_edges_difference_easy import create_edges
        create_fun = create_edges
    elif metric_type == "normal":
        from metric_functions.edge_creating.create_edges_normal import create_edges
        create_fun = lambda tree_param: create_edges(gold_text, tree_param)
    else:
        print(f"Error metric_type: {metric_type}")
        return None, None, None

    pred_unlabeled_edges, pred_labeled_edges = create_fun(pred_tree)
    pred_labeled_edges = [(e[0], e[1], e[2].split(":")[0]) for e in pred_labeled_edges]

    #print(pred_labeled_edges_set)
    # .split(":")[0] - from src.sentence_utils.simplify_relations

    gold_unlabeled_edges, gold_labeled_edges = create_fun(gold_tree)
    gold_labeled_edges = [(e[0], e[1], e[2].split(":")[0]) for e in gold_labeled_edges]
    return pred_unlabeled_edges, gold_unlabeled_edges, pred_labeled_edges, gold_labeled_edges

def calculate_f1(pred_eges, gold_edges):
    pred_len = len(pred_eges)
    gold_len = len(gold_edges)
    numerator = len(Counter(pred_eges) & Counter(gold_edges)) # Числитель

    precision = numerator / pred_len
    recall = numerator / gold_len
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1

def calculate_coeffs(pred_unlabeled_edges, gold_unlabeled_edges, pred_labeled_edges, gold_labeled_edges, uas, las, metric_type):
    pred_nodes = {e[0] for e in pred_unlabeled_edges} # Для LLM-версии это множество форм слов
    gold_nodes = {e[0] for e in gold_unlabeled_edges}

    g_not_p = len([e for e in gold_unlabeled_edges if e[0] not in pred_nodes])
    p_not_g = len([e for e in pred_unlabeled_edges if e[0] not in gold_nodes])

    las_numerator = len(Counter(pred_labeled_edges) & Counter(gold_labeled_edges)) # Числитель
    uas_numerator = len(Counter(pred_unlabeled_edges) & Counter(gold_unlabeled_edges)) # Числитель


    pred_len = len(pred_unlabeled_edges)
    gold_len = len(gold_unlabeled_edges)

    g_p_h_m_r = las_numerator
    g_p_h_m_not_r = uas_numerator - las_numerator
    if metric_type != "difference_easy": # TODO: пока так определяем не LLM вариант
        g_p_not_h = len([e for e in gold_unlabeled_edges
            if e[0] in pred_nodes and e[1] not in pred_nodes and "root" not in e[1]])
        right_tok_val = gold_len - g_not_p - g_p_not_h
        right_tok_coeff = right_tok_val
        synt_fun = lambda x: (x / right_tok_coeff) if right_tok_coeff != 0 else None
        tok_part = ((g_not_p + p_not_g + 2 * g_p_not_h) / (2 * right_tok_coeff)) if right_tok_coeff != 0 else 0
    else:

        g_p_hp_not_hg = len([e for e in pred_unlabeled_edges
            if e[0] in gold_nodes and e[1] not in gold_nodes and "root" not in e[1]])
        g_p_hg_not_hp = len([e for e in gold_unlabeled_edges
            if e[0] in pred_nodes and e[1] not in pred_nodes and "root" not in e[1]])
        right_tok_val_g = gold_len - g_not_p - g_p_hg_not_hp
        right_tok_val_p = pred_len - p_not_g - g_p_hp_not_hg
        right_tok_coeff = right_tok_val_g + right_tok_val_p
        synt_fun = lambda x: (2 * x / right_tok_coeff) if right_tok_coeff != 0 else None
        tok_part = ((g_not_p + g_p_hg_not_hp + p_not_g + g_p_hp_not_hg) / right_tok_coeff) if right_tok_coeff != 0 else 0

    if right_tok_coeff != 0:
        tok_coeff = 1 / (1 + tok_part)
        unlab_coeff = synt_fun(g_p_h_m_not_r + g_p_h_m_r)
        lab_coeff = synt_fun(g_p_h_m_r)
        if round(tok_coeff * unlab_coeff - uas, 5) != 0 or \
            round(tok_coeff * lab_coeff - las, 5) != 0:
            print(f"Error coeffs. uas: {uas}, las: {las}, coeff: {coeff_dict}")
    else:
        tok_coeff = 0
        unlab_coeff, lab_coeff = None, None

    coeff_dict = { "tok_coeff": tok_coeff, "unlab_coeff": unlab_coeff, "lab_coeff": lab_coeff}

    return coeff_dict

def create_statistics(gold_text, gold_tree, pred_tree, metric_type):
    pred_unlabeled_edges, gold_unlabeled_edges, pred_labeled_edges, gold_labeled_edges = process_trees(
         gold_text, gold_tree, pred_tree, metric_type)
    uas = calculate_f1(pred_unlabeled_edges, gold_unlabeled_edges)
    las = calculate_f1(pred_labeled_edges, gold_labeled_edges)
    coeff_dict = calculate_coeffs(pred_unlabeled_edges, gold_unlabeled_edges,
        pred_labeled_edges, gold_labeled_edges, uas, las, metric_type)

    return uas, las, coeff_dict
     