from metric_functions.difference_tokens.token_aligning import create_edges
from metric_functions.difference_tokens.data_preparing import delete_point_nodes
    
from collections import Counter
from copy import deepcopy

def create_statistics(gold_tree1_conll, output_tree1):
    gold_tree1 = [{'id': str(t['id']), 'form': t['form'],
               'parent_id': str(t['head']), 'relation': t['deprel'],
               'pos': t['upos'], 'feats': t['feats']} for t in gold_tree1_conll]
    gold_tree1 = delete_point_nodes(gold_tree1)

    if isinstance(output_tree1["pred_tree"], list):
        pred_tree = delete_point_nodes(output_tree1["pred_tree"])
        # TODO: delete_point_nodes - в предобработку. Еще нужен сдвиг вершин.
        pred_unlabeled_edges, pred_labeled_edges = create_edges(pred_tree)
        pred_labeled_edges = [(e[0], e[1], e[2].split(":")[0]) for e in pred_labeled_edges]


        #print(pred_labeled_edges_set)
        # .split(":")[0] - from src.sentence_utils.simplify_relations

        gold_unlabeled_edges, gold_labeled_edges = create_edges(gold_tree1)
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
    else:
        return None, None # TODO: Для точек Stanza
