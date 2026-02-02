from collections import Counter
from copy import deepcopy

def create_statistics(gold_text, gold_tree, pred_tree, metric_type):
    if metric_type == "difference":
        from metric_functions.edge_creating.creating_edges_difference import create_edges
        create_fun = create_edges
    else:
        from metric_functions.edge_creating.create_edges_normal import create_edges
        create_fun = lambda tree_param: create_edges(gold_text, tree_param)

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

    return uas, las