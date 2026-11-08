from metric_functions.category_calculating \
            import create_statistics


def calculate_mean_metrics(uas_metrics, las_metrics):
    good_uas = [r for r in uas_metrics if r is not None]
    good_las = [r for r in las_metrics if r is not None]
    mean_res = {}
    mean_res["uas_right"] = sum(good_uas) / len(good_uas) if len(good_uas) > 0 else 0
    mean_res["uas_all"] = sum(good_uas) / len(uas_metrics)
    mean_res["las_right"] = sum(good_las) / len(good_las) if len(good_las) > 0 else 0
    mean_res["las_all"] = sum(good_las) / len(las_metrics)
    mean_res["wrong_amount"] = len(uas_metrics) - len(good_uas)
    mean_res["all_amount"] = len(uas_metrics)
    return mean_res

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
