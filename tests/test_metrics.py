from src.eval.metrics import (
    BACKGROUND_LABEL,
    MISSED_LABEL,
    confusion_matrix,
    evaluate,
    per_class_metrics,
    precision_recall_f1,
    temporal_iou,
)


def seg(start, end, label="a", confidence=1.0):
    return {"start": start, "end": end, "label": label, "confidence": confidence}


def test_temporal_iou_full_overlap():
    a = seg(0, 10)
    assert temporal_iou(a, a) == 1.0


def test_temporal_iou_no_overlap():
    assert temporal_iou(seg(0, 5), seg(10, 15)) == 0.0


def test_temporal_iou_partial_overlap():
    # intersection [5,10]=5, union [0,10]+[5,15]-5=15
    assert abs(temporal_iou(seg(0, 10), seg(5, 15)) - 5 / 15) < 1e-9


def test_temporal_iou_adjacent_segments_no_overlap():
    assert temporal_iou(seg(0, 5), seg(5, 10)) == 0.0


def test_precision_recall_f1_perfect_match():
    preds = [seg(0, 10, "fighting")]
    gts = [seg(0, 10, "fighting")]
    m = precision_recall_f1(preds, gts, 0.5)
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_precision_recall_f1_wrong_label_is_fp_and_fn():
    preds = [seg(0, 10, "fighting")]
    gts = [seg(0, 10, "vandalism")]
    m = precision_recall_f1(preds, gts, 0.5)
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 1


def test_precision_recall_f1_below_iou_threshold_not_matched():
    preds = [seg(0, 2, "fighting")]  # tiny overlap with [0,10]
    gts = [seg(0, 10, "fighting")]
    m = precision_recall_f1(preds, gts, 0.5)
    assert m["tp"] == 0
    assert m["fp"] == 1 and m["fn"] == 1


def test_matching_is_one_to_one_not_many_to_one():
    # two predictions both overlapping one GT segment should yield only 1 TP
    preds = [seg(0, 10, "fighting"), seg(1, 9, "fighting")]
    gts = [seg(0, 10, "fighting")]
    m = precision_recall_f1(preds, gts, 0.5)
    assert m["tp"] == 1
    assert m["fp"] == 1  # the second, unmatched prediction
    assert m["fn"] == 0


def test_matching_prefers_highest_iou_pair():
    # gt2 overlaps pred more tightly than gt1 does; only one can be matched (one-to-one)
    preds = [seg(0, 10, "fighting")]
    gts = [seg(0, 3, "fighting"), seg(0, 10, "fighting")]
    m = precision_recall_f1(preds, gts, 0.5)
    assert m["tp"] == 1 and m["fn"] == 1  # one gt matched, one left over


def test_empty_predictions_all_false_negatives():
    gts = [seg(0, 10, "fighting"), seg(20, 30, "loitering")]
    m = precision_recall_f1([], gts, 0.5)
    assert m["tp"] == 0 and m["fn"] == 2 and m["precision"] == 0.0


def test_empty_ground_truth_all_false_positives():
    preds = [seg(0, 10, "fighting")]
    m = precision_recall_f1(preds, [], 0.5)
    assert m["tp"] == 0 and m["fp"] == 1 and m["recall"] == 0.0


def test_per_class_metrics_isolates_classes():
    preds = [seg(0, 10, "fighting"), seg(20, 30, "loitering")]
    gts = [seg(0, 10, "fighting"), seg(20, 30, "vandalism")]
    result = per_class_metrics(preds, gts, 0.5)
    assert result["fighting"]["tp"] == 1
    assert result["loitering"]["fp"] == 1
    assert result["vandalism"]["fn"] == 1


def test_confusion_matrix_counts_background_and_missed():
    preds = [seg(0, 10, "fighting"), seg(50, 60, "vandalism")]  # second pred has no overlapping gt
    gts = [seg(0, 10, "fighting"), seg(20, 30, "loitering")]  # second gt has no overlapping pred
    cm = confusion_matrix(preds, gts, localization_iou_threshold=0.3)
    assert cm["fighting"]["fighting"] == 1
    assert cm["loitering"][MISSED_LABEL] == 1
    assert cm[BACKGROUND_LABEL]["vandalism"] == 1


def test_evaluate_sweeps_multiple_thresholds():
    preds = [seg(0, 10, "fighting")]
    gts = [seg(0, 10, "fighting")]
    result = evaluate(preds, gts, iou_thresholds=(0.3, 0.5, 0.7))
    assert set(result["overall"].keys()) == {"0.3", "0.5", "0.7"}
    assert result["overall"]["0.5"]["tp"] == 1
