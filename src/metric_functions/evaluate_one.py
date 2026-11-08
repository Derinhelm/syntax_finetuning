from metric_functions.category_calculating \
            import create_statistics

def evaluate_one_experiment(data, metric_type):
    expir_res_uas, expir_res_las, expir_res_coeffs = [], [], []
    for sent_i, d in enumerate(data):
        try:
            gold_tree = d["gold_tree"]
            pred_tree = d["pred_tree"]
            gold_text = d["gold_text"]
            if gold_tree is not None and pred_tree is not None:
                sent_uas, sent_las = create_statistics(
                    gold_text, gold_tree, pred_tree, metric_type)
            else: # Предложение с некорректным результатом
                sent_uas, sent_las = None, None, None
        except Exception as e:
            print(sent_i, e)
            sent_uas, sent_las = None, None, None
        expir_res_uas.append(sent_uas)
        expir_res_las.append(sent_las)

    return expir_res_uas, expir_res_las
