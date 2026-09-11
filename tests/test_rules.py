from src.detection.detector import Track
from src.rules.engine import (
    abandoned_object_scores,
    build_track_history,
    density_scores,
    dwell_time_scores,
    falling_scores,
    velocity_scores,
    zone_occupancy_scores,
)
from src.rules.geometry import Zone

RESTRICTED_ZONE = [Zone("z1", "restricted", [(0, 0), (100, 0), (100, 100), (0, 100)])]
LOITER_ZONE = [Zone("z2", "loitering", [(0, 0), (100, 0), (100, 100), (0, 100)])]
NO_RUN_ZONE = [Zone("z3", "no_running", [(0, 0), (200, 0), (200, 200), (0, 200)])]
FRAME_SIZE = (200, 200)


def person(track_id, bbox, cls_name="person"):
    return Track(track_id, cls_name, bbox, 0.9)


def frames_from(track_id_bboxes_per_ts: list[tuple[float, list[tuple[int, tuple]]]]):
    """[(ts, [(track_id, bbox), ...]), ...] -> [(ts, [Track, ...]), ...]"""
    return [(ts, [person(tid, bbox) for tid, bbox in entries]) for ts, entries in track_id_bboxes_per_ts]


def test_zone_occupancy_triggers_after_min_duration():
    # person stays inside the restricted zone [0,100]x[0,100] for 2 seconds
    frames = frames_from([(0.0, [(1, (10, 10, 20, 20))]), (1.0, [(1, (10, 10, 20, 20))]), (2.0, [(1, (10, 10, 20, 20))])])
    history = build_track_history(frames)
    scores = zone_occupancy_scores(history, RESTRICTED_ZONE, FRAME_SIZE, FRAME_SIZE, min_trigger_duration_s=0.5, global_timestamps=[0.0, 1.0, 2.0])
    scores_dict = dict(scores)
    assert scores_dict[0.0] < 1.0  # not yet at the trigger duration
    assert scores_dict[2.0] == 1.0


def test_zone_occupancy_zero_outside_zone():
    frames = frames_from([(0.0, [(1, (150, 150, 160, 160))]), (1.0, [(1, (150, 150, 160, 160))])])
    history = build_track_history(frames)
    scores = zone_occupancy_scores(history, RESTRICTED_ZONE, FRAME_SIZE, FRAME_SIZE, min_trigger_duration_s=0.5, global_timestamps=[0.0, 1.0])
    assert all(s == 0.0 for _, s in scores)


def test_dwell_time_ramps_up_with_sustained_presence():
    ts_list = [float(i) for i in range(0, 20, 2)]  # 0,2,4,...,18 (well past the 15s threshold)
    frames = frames_from([(ts, [(1, (10, 10, 20, 20))]) for ts in ts_list])
    history = build_track_history(frames)
    scores = dwell_time_scores(history, LOITER_ZONE, FRAME_SIZE, FRAME_SIZE, dwell_threshold_s=15.0, score_sigmoid_scale_s=5.0, stillness_radius_px=60, global_timestamps=ts_list)
    scores_dict = dict(scores)
    assert scores_dict[0.0] < scores_dict[18.0]
    assert scores_dict[18.0] > 0.5  # past the threshold, sigmoid should be > 0.5


def test_dwell_time_resets_when_person_walks_away():
    # person stands still 0-8s, then walks far away and keeps moving — should
    # never accumulate enough continuous stillness to cross the threshold
    still_part = [(float(i), [(1, (10, 10, 20, 20))]) for i in range(0, 9)]
    walking_part = [(float(i), [(1, (10 + (i - 8) * 50, 10, 20 + (i - 8) * 50, 20))]) for i in range(9, 20)]
    frames = frames_from(still_part + walking_part)
    history = build_track_history(frames)
    ts_list = [ts for ts, _ in frames]
    scores = dwell_time_scores(history, LOITER_ZONE, FRAME_SIZE, FRAME_SIZE, dwell_threshold_s=15.0, score_sigmoid_scale_s=5.0, stillness_radius_px=60, global_timestamps=ts_list)
    assert dict(scores)[19.0] < 0.5  # never re-accumulates 15s of stillness after walking off


def test_velocity_scores_high_for_fast_sustained_movement():
    # centroid moves 20px every 0.2s (10 bbox-heights/s, well above the 2.0 threshold),
    # staying inside the [0,200]x[0,200] no_running zone the whole time
    ts_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    bboxes = [(10 + i * 20, 10, 20 + i * 20, 20) for i in range(len(ts_list))]
    frames = frames_from(list(zip(ts_list, [[(1, b)] for b in bboxes])))
    history = build_track_history(frames)
    scores = velocity_scores(history, NO_RUN_ZONE, FRAME_SIZE, FRAME_SIZE, velocity_threshold_bbox_heights_per_s=2.0, min_trigger_duration_s=0.5, global_timestamps=ts_list)
    assert dict(scores)[1.2] > 0.0


def test_velocity_scores_zero_for_stationary_person():
    ts_list = [0.0, 0.5, 1.0, 1.5]
    frames = frames_from([(ts, [(1, (10, 10, 20, 20))]) for ts in ts_list])
    history = build_track_history(frames)
    scores = velocity_scores(history, NO_RUN_ZONE, FRAME_SIZE, FRAME_SIZE, velocity_threshold_bbox_heights_per_s=2.0, min_trigger_duration_s=0.5, global_timestamps=ts_list)
    assert all(s == 0.0 for _, s in scores)


def test_density_detects_crowd_cluster():
    # 4 people clustered close together for 3+ seconds
    ts_list = [0.0, 1.0, 2.0, 3.0, 4.0]
    cluster = [(i, (i * 10, 0, i * 10 + 20, 20)) for i in range(4)]
    frames = frames_from([(ts, cluster) for ts in ts_list])
    scores = density_scores(frames, min_people=4, cluster_radius_px=100, sustained_duration_s=3.0)
    assert dict(scores)[4.0] == 1.0
    assert dict(scores)[0.0] < 1.0


def test_density_ignores_too_few_people():
    ts_list = [0.0, 1.0, 2.0]
    cluster = [(i, (i * 10, 0, i * 10 + 20, 20)) for i in range(2)]  # only 2 people
    frames = frames_from([(ts, cluster) for ts in ts_list])
    scores = density_scores(frames, min_people=4, cluster_radius_px=100, sustained_duration_s=3.0)
    assert all(s == 0.0 for _, s in scores)


def test_falling_detects_aspect_ratio_flip_then_stillness():
    # standing (tall, narrow) for a while, then flips to lying (wide, short) and stays still
    ts_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    standing = (10, 10, 20, 50)   # w=10 h=40 -> ar=0.25
    lying = (10, 40, 60, 55)      # w=50 h=15 -> ar=3.3, big flip vs standing
    bboxes = [standing, standing, standing, lying, lying, lying]
    frames = frames_from(list(zip(ts_list, [[(1, b)] for b in bboxes])))
    history = build_track_history(frames)
    scores = falling_scores(history, aspect_ratio_flip_ratio=1.4, post_event_stillness_window_s=2.0, post_event_movement_px=15, global_timestamps=ts_list)
    assert any(s == 1.0 for _, s in scores)


def test_falling_zero_when_no_shape_change():
    ts_list = [0.0, 1.0, 2.0, 3.0]
    frames = frames_from([(ts, [(1, (10, 10, 20, 50))]) for ts in ts_list])
    history = build_track_history(frames)
    scores = falling_scores(history, aspect_ratio_flip_ratio=1.4, post_event_stillness_window_s=2.0, post_event_movement_px=15, global_timestamps=ts_list)
    assert all(s == 0.0 for _, s in scores)


def test_abandoned_object_triggers_when_static_and_unowned():
    ts_list = [float(i) for i in range(0, 25, 1)]
    # a backpack that stays put the whole time, no person ever nearby
    frames = [(ts, [Track(1, "backpack", (10, 10, 30, 30), 0.9)]) for ts in ts_list]
    history = build_track_history(frames)
    scores = abandoned_object_scores(
        frames, history, static_object_classes=["backpack"], static_duration_s=10.0,
        static_movement_px=20, owner_proximity_px=50, owner_separation_duration_s=10.0,
        global_timestamps=ts_list,
    )
    assert dict(scores)[24.0] == 1.0


def test_abandoned_object_zero_when_owner_nearby():
    ts_list = [float(i) for i in range(0, 25, 1)]
    frames = [
        (ts, [Track(1, "backpack", (10, 10, 30, 30), 0.9), Track(2, "person", (15, 15, 35, 60), 0.9)])
        for ts in ts_list
    ]
    history = build_track_history(frames)
    scores = abandoned_object_scores(
        frames, history, static_object_classes=["backpack"], static_duration_s=10.0,
        static_movement_px=20, owner_proximity_px=50, owner_separation_duration_s=10.0,
        global_timestamps=ts_list,
    )
    assert all(s == 0.0 for _, s in scores)
