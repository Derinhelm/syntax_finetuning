from collections import Counter
import json

def create_edges(tree):
    counter = Counter()
    id_dict = {}
    for t in tree:
      if "." not in t["id"]:
        t_form = t["form"]
        t_id = t["id"]
        cur_val = f"{t_form}_{counter[t_form]}"
        id_dict[t_id] = cur_val
        counter.update([t_form])
    id_dict["0"] = "root_0"
    unlabeled_edges, labeled_edges = [], []
    for t in tree:
      if "." not in t["id"]:
        parent_node = id_dict.get(t["parent_id"], "None")
        node = id_dict[t["id"]]
        unlabeled_edges.append((node, parent_node))
        labeled_edges.append((node, parent_node, t['relation']))
    unlabeled_edges_set = set(unlabeled_edges)
    labeled_edges_set = set(labeled_edges)
    assert len(unlabeled_edges) == len(unlabeled_edges_set)
    assert len(labeled_edges) == len(labeled_edges_set)
    assert len(unlabeled_edges_set) == len(labeled_edges_set)
    return unlabeled_edges_set, labeled_edges_set

def calculate_sent_metrucs(sent):
    if not isinstance(sent['pred_tree'], list):
        return None, None
    pred_unlab_edges, pred_lab_edges = create_edges(sent['pred_tree'])
    gold_unlab_edges, gold_lab_edges = create_edges(sent['gold_tree'])
    uas_numerator = len(pred_unlab_edges & gold_unlab_edges)
    las_numerator = len(pred_lab_edges & gold_lab_edges)
    prec_denominator = len(pred_unlab_edges)
    recall_denominator = len(gold_unlab_edges)
    uas_prec = uas_numerator / prec_denominator
    uas_recall = uas_numerator / recall_denominator
    las_prec = las_numerator / prec_denominator
    las_recall = las_numerator / recall_denominator
    if uas_prec + uas_recall > 0:
        uas_f = (2 * uas_prec * uas_recall) / (uas_prec + uas_recall)
    else:
        uas_f = 0.0
    if las_prec + las_recall > 0:
        las_f = (2 * las_prec * las_recall) / (las_prec + las_recall)
    else:
        las_f = 0.0
    assert uas_f >= las_f
    return uas_f, las_f


def calculate(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    res = [calculate_sent_metrucs(sent) for sent in data]
    return (res)

res = calculate('../../pred_results/Qwen06_Instruct_grct_syntagrus.json') # TODO
good_uas = [r[0] for r in res if r[0] is not None]
good_las = [r[1] for r in res if r[1] is not None]
print(sum(good_uas) / len(good_uas))
print(sum(good_las) / len(good_las))

