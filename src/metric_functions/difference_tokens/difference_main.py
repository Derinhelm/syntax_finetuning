from metric_functions.difference_tokens.category_calculating \
            import create_statistics

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

# TODO: убрать лишнюю функцию
def process_sentence(gold_tree, parser_tree):
    sent_uas, sent_las = create_statistics(gold_tree, parser_tree)
    return sent_uas, sent_las
