from metric_functions.difference_tokens.metric_calculating import calculate_metrics
from metric_functions.difference_tokens.category_calculating import create_statistics

def print_mean_metrics(uas_metrics, las_metrics):
    good_uas = [r for r in uas_metrics if r is not None]
    bad_uas = [r for r in uas_metrics if r is None]
    good_las = [r for r in las_metrics if r is not None]
    bad_las = [r for r in las_metrics if r is None]
    if good_uas:
        print(f"{sum(good_uas) / len(good_uas) * 100:.1f}% ({sum(good_uas) / len(uas_metrics) * 100:.1f}%)")
    if good_las:
        print(f"{sum(good_las) / len(good_las) * 100:.1f}% ({sum(good_las) / len(las_metrics) * 100:.1f}%)")
    print(len(bad_uas), len(uas_metrics))#len(bad_las))
    
def process_sentence(gold_tree, parser_tree):
    output_r, output_errors, output_extra_pred_tokens = create_statistics(
        gold_tree, parser_tree)
    # TODO: output_errors и output_extra_pred_tokens - для сохранения
    sent_uas, sent_las = \
        calculate_metrics(output_r)
    return sent_uas, sent_las
