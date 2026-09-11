from src.pipeline.segment_builder import build_all_segments, build_segments_for_class, moving_median


def test_moving_median_smooths_single_spike():
    values = [0.1, 0.1, 0.9, 0.1, 0.1]
    smoothed = moving_median(values, window=3)
    assert smoothed[2] < 0.9  # the isolated spike gets suppressed


def test_moving_median_noop_for_window_1():
    values = [0.1, 0.5, 0.9]
    assert moving_median(values, window=1) == values


def test_build_segments_for_class_basic_threshold():
    series = [(0.0, 0.1), (1.0, 0.9), (2.0, 0.9), (3.0, 0.1)]
    segments = build_segments_for_class(
        series, interval_s=1.0, label="Fighting", threshold=0.5,
        smoothing_window=1, min_duration_s=0.5, merge_gap_s=0.5,
    )
    assert len(segments) == 1
    assert segments[0]["label"] == "Fighting"
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 3.0  # last active sample (2.0) + interval (1.0)


def test_build_segments_filters_short_segments():
    series = [(0.0, 0.1), (1.0, 0.9), (2.0, 0.1)]
    segments = build_segments_for_class(
        series, interval_s=1.0, label="Fighting", threshold=0.5,
        smoothing_window=1, min_duration_s=5.0, merge_gap_s=0.5,
    )
    assert segments == []


def test_build_segments_merges_close_runs():
    # two active runs separated by a 1-sample gap, within merge_gap_s
    series = [(0.0, 0.9), (1.0, 0.1), (2.0, 0.9), (3.0, 0.9)]
    segments = build_segments_for_class(
        series, interval_s=1.0, label="Fighting", threshold=0.5,
        smoothing_window=1, min_duration_s=0.5, merge_gap_s=1.5,
    )
    assert len(segments) == 1
    assert segments[0]["start"] == 0.0


def test_build_segments_does_not_merge_far_apart_runs():
    series = [(0.0, 0.9), (1.0, 0.1), (10.0, 0.9)]
    segments = build_segments_for_class(
        series, interval_s=1.0, label="Fighting", threshold=0.5,
        smoothing_window=1, min_duration_s=0.5, merge_gap_s=1.0,
    )
    assert len(segments) == 2


def test_build_all_segments_allows_cross_class_overlap():
    class_series = {
        "fighting": [(0.0, 0.9), (1.0, 0.9)],
        "crowd_formation": [(0.0, 0.9), (1.0, 0.9)],
    }
    class_intervals = {"fighting": 1.0, "crowd_formation": 1.0}
    class_display_names = {"fighting": "Fighting", "crowd_formation": "Crowd Formation"}
    post_cfg = {"confidence_threshold": 0.5, "smoothing_window_frames": 1, "min_segment_duration_s": 0.5, "merge_gap_s": 0.5}
    segments = build_all_segments(class_series, class_intervals, class_display_names, post_cfg)
    labels = {s["label"] for s in segments}
    assert labels == {"Fighting", "Crowd Formation"}
    # both segments occupy the same time range — overlap is allowed, not deduplicated
    assert all(s["start"] == 0.0 for s in segments)


def test_build_segments_sorted_by_start_time():
    class_series = {"a": [(5.0, 0.9)], "b": [(1.0, 0.9)]}
    class_intervals = {"a": 1.0, "b": 1.0}
    class_display_names = {"a": "A", "b": "B"}
    post_cfg = {"confidence_threshold": 0.5, "smoothing_window_frames": 1, "min_segment_duration_s": 0.0, "merge_gap_s": 0.0}
    segments = build_all_segments(class_series, class_intervals, class_display_names, post_cfg)
    assert [s["label"] for s in segments] == ["B", "A"]
