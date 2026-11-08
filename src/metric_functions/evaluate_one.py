import json
import gc
from conllu import parse

from metric_functions.category_calculating \
            import create_statistics
from metric_functions.data_preparing import preprocess_tree


def calculate_mean_metrics(uas_metrics, las_metrics, coeffs):
    good_uas = [r for r in uas_metrics if r is not None]
    good_las = [r for r in las_metrics if r is not None]
    mean_res = {}
    mean_res["uas_right"] = sum(good_uas) / len(good_uas) if len(good_uas) > 0 else 0
    mean_res["uas_all"] = sum(good_uas) / len(uas_metrics)
    mean_res["las_right"] = sum(good_las) / len(good_las) if len(good_las) > 0 else 0
    mean_res["las_all"] = sum(good_las) / len(las_metrics)
    mean_res["wrong_amount"] = len(uas_metrics) - len(good_uas)
    mean_res["all_amount"] = len(uas_metrics)

    token_coeffs = [c_dict["tok_coeff"] if c_dict is not None else 0.0 for c_dict in coeffs]
    uas_parser_coeffs = [c_dict["unlab_coeff"] if c_dict is not None and c_dict["unlab_coeff"] is not None else 0
                    for c_dict in coeffs]
    las_parser_coeffs = [c_dict["lab_coeff"] if c_dict is not None and c_dict["lab_coeff"] is not None else 0
                    for c_dict in coeffs]

    mean_res["mean_tok_coeffs"] = sum(token_coeffs) / len(token_coeffs)
    mean_res["mean_uas_coeffs"] = sum(uas_parser_coeffs) / len(uas_parser_coeffs)
    mean_res["mean_las_coeffs"] = sum(las_parser_coeffs) / len(las_parser_coeffs)

    return mean_res

def evaluate_one_experiment(gold_sentences, pred_trees, metric_type):

    expir_res_uas, expir_res_las, expir_res_coeffs = [], [], []
    pred_trees_dict = {tree['index']:tree for tree in pred_trees}
    assert len(pred_trees_dict) == len(pred_trees)
    for sent_i, sent_r in enumerate(gold_sentences):
        try:
            if isinstance(pred_trees_dict[sent_i]["pred_tree"], list):
                gold_tree = [{'id': str(t['id']), 'form': t['form'],
                    'parent_id': str(t['head']), 'relation': t['deprel'],
                    'pos': t['upos'], 'feats': t['feats']}
                        for t in gold_sentences[sent_i]]
                gold_text = gold_sentences[sent_i].metadata['text']
                gold_tree = preprocess_tree(gold_tree)
                pred_tree = preprocess_tree(
                    pred_trees_dict[sent_i]["pred_tree"])
                sent_uas, sent_las, sent_coeff_dict = create_statistics(
                    gold_text, gold_tree, pred_tree, metric_type)
            else: # Предложение с некорректным результатом
                sent_uas, sent_las, sent_coeff_dict = None, None, None

        except Exception as e:
            print(sent_i, e)
            sent_uas, sent_las, sent_coeff_dict = None, None, None
        expir_res_uas.append(sent_uas)
        expir_res_las.append(sent_las)
        if sent_coeff_dict is not None:
            sent_coeff_dict["index"] = sent_i
        expir_res_coeffs.append(sent_coeff_dict)

    del pred_trees
    gc.collect()
    return expir_res_uas, expir_res_las, expir_res_coeffs
