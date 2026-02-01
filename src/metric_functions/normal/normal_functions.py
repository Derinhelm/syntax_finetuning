def create_sent_be_nodes(sent_text, sent_tokens, text_transform):
    token_begin_end = []
    begin_end_token_dict = {}
    sent_text = text_transform(sent_text)
    original_sent_text = text_transform(sent_text)
    del_prefix_len = 0
    tokens = [t for t in sent_tokens if '.' not in t["id"]]
    for t_i, t in enumerate(tokens):
        token_text = text_transform(t["form"])
        t_start = sent_text.find(token_text)
        if t_start == -1:
            print(f"Error. sent_text:{sent_text}, t:{token_text}", t_i)
        else:
            b, e = (del_prefix_len + t_start,
                                  del_prefix_len + t_start + len(token_text))
            token_begin_end.append((t, (b, e)))
            begin_end_token_dict[(b, e)] = t
            del_prefix_len += t_start + len(token_text)
            sent_text = sent_text[t_start + len(token_text):]
            assert text_transform(original_sent_text[b:e]) == \
                text_transform(tokens[t_i]["form"])
    sent_text = text_transform(sent_text)
    return token_begin_end, begin_end_token_dict

def create_sent_be_edges(sent_be_tokens):
    sent_be_res = {}
    for t_id, (t, t_be) in enumerate(sent_be_tokens): # ellipsis are deleted, so index in sent_be_tokens = token_id
      parent_id = t["parent_id"]
      if parent_id == '0': # root
        parent_be = (-1, -1)
      else:
        _, parent_be = sent_be_tokens[int(parent_id) - 1]
      sent_be_res[t_be] = (parent_be, t["relation"])
    return sent_be_res
