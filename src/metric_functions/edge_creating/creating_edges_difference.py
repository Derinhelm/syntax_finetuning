import argparse
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
    assert len(unlabeled_edges) == len(labeled_edges)
    return unlabeled_edges, labeled_edges
