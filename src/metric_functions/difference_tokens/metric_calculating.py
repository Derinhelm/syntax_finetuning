  
def calculate_metrics(sent_res):
    if sent_res["categories"] is not None:
        uas_precision = (sent_res["categories"][3] + sent_res["categories"][4]) / sent_res["pred_len"]
        uas_recall = (sent_res["categories"][3] + sent_res["categories"][4]) / sent_res["gold_len"]
        if uas_precision + uas_recall > 0:
            uas = (2 * uas_precision * uas_recall) / (uas_precision + uas_recall)
        else:
            uas = 0.0
        las_precision = sent_res["categories"][4] / sent_res["pred_len"]
        las_recall = sent_res["categories"][4] / sent_res["gold_len"]
        if (las_precision + las_recall) > 0:
            las = (2 * las_precision * las_recall) / (las_precision + las_recall)
        else:
            las = 0.0
        return uas, las
    else:
        return None, None