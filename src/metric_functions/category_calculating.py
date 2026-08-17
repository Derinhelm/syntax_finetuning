from collections import Counter
from copy import deepcopy

def create_statistics(gold_text, gold_tree, pred_tree, metric_type):
    if metric_type == "difference":
        from metric_functions.edge_creating.creating_edges_difference import create_edges
        create_fun = create_edges
    elif metric_type == "difference_easy":
        from metric_functions.edge_creating.creating_edges_difference_easy import create_edges
        create_fun = create_edges
    elif metric_type == "normal":
        from metric_functions.edge_creating.create_edges_normal import create_edges
        create_fun = lambda tree_param: create_edges(gold_text, tree_param)
    else:
        print(f"Error metric_type: {metric_type}")
        return None, None

    pred_unlabeled_edges, pred_labeled_edges = create_fun(pred_tree)
    pred_labeled_edges = [(e[0], e[1], e[2].split(":")[0]) for e in pred_labeled_edges]

    #print(pred_labeled_edges_set)
    # .split(":")[0] - from src.sentence_utils.simplify_relations

    gold_unlabeled_edges, gold_labeled_edges = create_fun(gold_tree)
    gold_labeled_edges = [(e[0], e[1], e[2].split(":")[0]) for e in gold_labeled_edges]

    pred_len = len(pred_unlabeled_edges)
    gold_len = len(gold_unlabeled_edges)

    uas_numerator = len(Counter(pred_unlabeled_edges) &
                        Counter(gold_unlabeled_edges)) # Числитель
    las_numerator = len(Counter(pred_labeled_edges) &
                        Counter(gold_labeled_edges)) # Числитель

    uas_precision = uas_numerator / pred_len
    uas_recall = uas_numerator / gold_len

    las_precision = las_numerator / pred_len
    las_recall = las_numerator / gold_len

    if uas_precision + uas_recall > 0:
        uas = (2 * uas_precision * uas_recall) / (uas_precision + uas_recall)
    else:
        uas = 0.0

    if (las_precision + las_recall) > 0:
        las = (2 * las_precision * las_recall) / (las_precision + las_recall)
    else:
        las = 0.0

    pred_nodes = {e[0] for e in pred_unlabeled_edges}
    gold_nodes = {e[0] for e in gold_unlabeled_edges}

    g_not_p = len([e for e in gold_unlabeled_edges if e[0] not in pred_nodes])
    p_not_g = len([e for e in pred_unlabeled_edges if e[0] not in gold_nodes])
    g_p_not_h = len([e for e in gold_unlabeled_edges
        if e[0] in pred_nodes and e[1] not in pred_nodes and "root" not in e[1]])

    g_p_h_m_r = las_numerator
    g_p_h_m_not_r = uas_numerator - las_numerator
    g_p_h_not_m = gold_len - g_not_p - g_p_not_h - g_p_h_m_r  - g_p_h_m_not_r

    if g_p_h_not_m + g_p_h_m_not_r + g_p_h_m_r != 0:
        tok_coeff = 1 / (1 + (g_not_p + p_not_g + 2 * g_p_not_h) / (2 * (g_p_h_not_m + g_p_h_m_not_r + g_p_h_m_r)))
        unlab_coeff = (g_p_h_m_not_r + g_p_h_m_r) / (g_p_h_not_m + g_p_h_m_not_r + g_p_h_m_r)
        lab_coeff = g_p_h_m_r / (g_p_h_not_m + g_p_h_m_not_r + g_p_h_m_r)
        if round(tok_coeff * unlab_coeff - uas, 5) != 0 or \
            round(tok_coeff * lab_coeff - las, 5) != 0:
            print(f"Error coeffs. uas: {uas}, las: {las}, coeff: {coeff_dict}")
    else:
        tok_coeff = 0
        unlab_coeff, lab_coeff = None, None

    coeff_dict = { "tok_coeff": tok_coeff, "unlab_coeff": unlab_coeff, "lab_coeff": lab_coeff}

    return uas, las, coeff_dict