from metric_functions.category_calculating \
            import create_statistics

def evaluate_one_experiment(data, metric_type, metrics):
    res = {m["name"]: [] for m in metrics}
    for sent_i, d in enumerate(data):
        try:
            gold_tree = d["gold_tree"]
            pred_tree = d["pred_tree"]
            gold_text = d["gold_text"]
            if gold_tree is not None and pred_tree is not None:
                sent_res = create_statistics(metrics,
                    gold_text, gold_tree, pred_tree, metric_type)
            else: # Предложение с некорректным результатом
                sent_res = {m["name"]: None for m in metrics}
        except Exception as e:
            print(sent_i, e)
            sent_res = {m["name"]: None for m in metrics}
        for m_name in res:
            res[m_name] = sent_res[m_name]
    return res
