import traceback

from metric_functions.category_calculating \
            import create_statistics

def evaluate_one_experiment(data, metric_type, metrics):
    res = {m["name"]: [] for m in metrics}
    for sent_i, d in enumerate(data):
        try:
            sent_res = create_statistics(metrics, d, metric_type)
        except Exception as e:
            print(sent_i, e)
            print(traceback.print_exc())
            sent_res = {m["name"]: None for m in metrics}
        for m_name in res:
            res[m_name].append({"index": sent_i, "metric": sent_res[m_name]})
    return res
