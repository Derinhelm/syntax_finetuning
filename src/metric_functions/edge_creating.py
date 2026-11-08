
def create_sent_be_nodes(sent_text, sent_tokens):
    # TODO: For DeepPavlov
    # transform_fun = lambda text: text.lower().replace('``', '"').replace("''", '"')
    text_transform = lambda text: text.lower().replace("''", '"')
    token_begin_end = []
    begin_end_token_dict = {}
    id_be_dict = {"0": (-1, -1)}
    sent_text = text_transform(sent_text)
    original_sent_text = text_transform(sent_text)
    del_prefix_len = 0
    tokens = [t for t in sent_tokens
              if "." not in t["id"] and "-" not in t['id']]
    for t_i, t in enumerate(tokens):
        token_text = text_transform(t["form"])
        t_start = sent_text.find(token_text)
        if t_start == -1:
            print(f"Error. sent_text:{sent_text}, t:{token_text}, original_sent_text:{original_sent_text}", t_i)
        else:
            b, e = (del_prefix_len + t_start,
                                  del_prefix_len + t_start + len(token_text))
            token_begin_end.append((t, (b, e)))
            begin_end_token_dict[(b, e)] = t
            del_prefix_len += t_start + len(token_text)
            sent_text = sent_text[t_start + len(token_text):]
            assert text_transform(original_sent_text[b:e]) == \
                text_transform(tokens[t_i]["form"])
            t_id = t["id"]
            id_be_dict[t_id] = (b, e)
    sent_text = text_transform(sent_text)
    return id_be_dict
    #return token_begin_end, begin_end_token_dict, id_be_dict

def create_form_nodes(gold_text, tree):
    id_form = {}
    for t in tree:
        t_id = t["id"]
        id_form[t_id] = t["form"]
    id_form["0"] = "root"
    return id_form

def create_edges_by_dict(tree, id_form, additional_fun = None):
    edges = []
    for t in tree:
        parent_node_form = id_form.get(t["parent_id"], "None")
        node_form = id_form[t["id"]]
        if additional_fun is None:
            edge_res = (node_form.strip(), parent_node_form.strip())
        else:
            edge_res = (node_form.strip(), parent_node_form.strip(), additional_fun(t))
        edges.append(edge_res)
    return edges

def create_nodes(gold_text, tree, metric_type):
    if metric_type == "difference_easy":
        create_node_fun = create_form_nodes
    elif metric_type == "normal":
        create_node_fun = create_sent_be_nodes
    else:
        print(f"Error metric_type: {metric_type}")
        return None, None, None
    id_value_dict = create_node_fun(gold_text, tree)
    return id_value_dict
