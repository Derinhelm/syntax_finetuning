import copy

def delete_point_nodes(sent_dict):
    return [t for t in sent_dict if "." not in t["id"] and "-" not in t["id"]]

def shift_token_id(tokens_param):
    tokens = copy.deepcopy(tokens_param)
    first_token_shift = 0
    for i, t in enumerate(tokens):
        if t["id"] == '1':
            first_token_shift = i
        shift_id = str(int(t["id"]) + first_token_shift)
        if t["parent_id"] != '0':
            shift_parent_id = str(int(t["parent_id"]) + first_token_shift)
        else:
            shift_parent_id = '0'
        t["id"] = shift_id
        t["parent_id"] = shift_parent_id
    return tokens

def preprocess_tree(sent_dict):
    sent_dict = delete_point_nodes(sent_dict)
    sent_dict = shift_token_id(sent_dict) # TODO: shift нельзя делать для результата, сгенерированного недообученной LLM
    return sent_dict
