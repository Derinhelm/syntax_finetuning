from collections import Counter

from metric_functions.edge_creating import create_edges_by_dict, create_nodes

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

'''
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
'''
    
def create_statistics(metrics, gold_text, gold_tree, pred_tree, metric_type):
    pred_id_value_dict = create_nodes(gold_text, pred_tree, metric_type)
    gold_id_value_dict = create_nodes(gold_text, gold_tree, metric_type)

    res = {}
    for m in metrics:
        metric_name = m["name"]
        metric_edge_fun = m["metric_edge_fun"]
        pred_edges = create_edges_by_dict(pred_tree, pred_id_value_dict, metric_edge_fun)
        gold_edges = create_edges_by_dict(gold_tree, gold_id_value_dict, metric_edge_fun)
        metric_value = calculate_f1(pred_edges, gold_edges)
        res[metric_name] = metric_value
    return res
     