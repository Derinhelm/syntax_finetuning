def delete_point_nodes(sent_dict):
    return [t for t in sent_dict if "." not in t["id"]]

def preprocess_tree(sent_dict):
    sent_dict = delete_point_nodes(sent_dict)
    return sent_dict
