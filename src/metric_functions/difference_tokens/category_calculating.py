from metric_functions.difference_tokens.token_aligning import create_edges
from metric_functions.difference_tokens.data_preparing import delete_point_nodes
    
from collections import Counter
from copy import deepcopy

def create_statistics(gold_tree1_conll, output_tree1):
    gold_tree1 = [{'id': str(t['id']), 'form': t['form'],
               'parent_id': str(t['head']), 'relation': t['deprel'],
               'pos': t['upos'], 'feats': t['feats']} for t in gold_tree1_conll]
    gold_tree1 = delete_point_nodes(gold_tree1)

    res = {}
    errors = {}
    res["gold_tree"] = deepcopy(gold_tree1)
    res["sent_text"] = gold_tree1_conll.metadata["text"]
    res["sent_id"] = gold_tree1_conll.metadata["sent_id"]
    res["pred_tree"] = output_tree1["pred_tree"]
    res["gold_len"] = len(res["gold_tree"])
    extra_pred_tokens = None

    if isinstance(output_tree1["pred_tree"], list):

        output_tree1["pred_tree"] = delete_point_nodes(output_tree1["pred_tree"])

        _, _, pred_unlabeled_edges_set, pred_labeled_edges_set = create_edges(output_tree1["pred_tree"])
        pred_labeled_edges_set = {(e[0], e[1], e[2].split(":")[0]) for e in pred_labeled_edges_set}
        res["pred_unlabeled_edges_set"] = pred_unlabeled_edges_set
        res["pred_labeled_edges_set"] = pred_labeled_edges_set
        #print(pred_labeled_edges_set)
        # .split(":")[0] - from src.sentence_utils.simplify_relations

        pred_nodes_set = { r[0] for r in pred_labeled_edges_set }
        #print(pred_nodes_set)
        assert len(output_tree1["pred_tree"]) == len(pred_unlabeled_edges_set)
        assert len(output_tree1["pred_tree"]) == len(pred_labeled_edges_set)

        gold_unlabeled_edges, gold_labeled_edges, _, _ = create_edges(res["gold_tree"])
        gold_labeled_edges = [(e[0], e[1], e[2].split(":")[0]) for e in gold_labeled_edges]
        assert len(res["gold_tree"]) == len(gold_unlabeled_edges)
        assert len(res["gold_tree"]) == len(gold_labeled_edges)

        for t_i in range(len(res["gold_tree"])):
            assert res["gold_tree"][t_i]["form"] in gold_unlabeled_edges[t_i][0]
            assert res["gold_tree"][t_i]["form"] in gold_labeled_edges[t_i][0]
            res["gold_tree"][t_i]["unlab_edge"] = gold_unlabeled_edges[t_i]
            res["gold_tree"][t_i]["lab_edge"] = gold_labeled_edges[t_i]
            if res["gold_tree"][t_i]["unlab_edge"][0] not in pred_nodes_set:
                category = 1
            elif res["gold_tree"][t_i]["unlab_edge"] not in pred_unlabeled_edges_set:
                category = 2
            elif res["gold_tree"][t_i]["lab_edge"] not in pred_labeled_edges_set:
                category = 3
            else:
                category = 4
            res["gold_tree"][t_i]["category"] = category

        res["categories"] = Counter(t["category"] for t in res["gold_tree"])

        res["pred_len"] = len(pred_unlabeled_edges_set)

        extra_pred_tokens = [f"{t['id']}_{t['form']}" for t in output_tree1["pred_tree"]
                             if t["form"] not in output_tree1["input"]]
    else:
        res["pred_len"] = None
        res["categories"] = None
        errors["pred_output"] = output_tree1["pred_output"]
        errors["full_pred_output"] = output_tree1["full_pred_output"]
        errors["pred_tree"] = output_tree1["pred_tree"]
        errors["sent_id"] = gold_tree1_conll.metadata["sent_id"]
        errors["index"] = output_tree1["index"]

    return res, errors, extra_pred_tokens
