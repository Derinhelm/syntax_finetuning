import argparse
from collections import Counter
import json

def create_edges(tree):
    counter = Counter()
    id_form = {}
    for t in tree:
        t_id = t["id"]
        id_form[t_id] = t["form"]
    id_form["0"] = "root"
    unlabeled_edges, labeled_edges = [], []
    for t in tree:
        parent_node_form = id_form.get(t["parent_id"], "None")
        node_form = id_form[t["id"]]
        unlabeled_edges.append((node_form, parent_node_form))
        labeled_edges.append((node_form, parent_node_form, t['relation']))
    assert len(unlabeled_edges) == len(labeled_edges)
    return unlabeled_edges, labeled_edges
