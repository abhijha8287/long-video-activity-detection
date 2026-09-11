from src.rules.geometry import euclidean, point_in_polygon, scale_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_inside_polygon():
    assert point_in_polygon((5, 5), SQUARE) is True


def test_point_outside_polygon():
    assert point_in_polygon((15, 5), SQUARE) is False


def test_point_outside_polygon_negative_coords():
    assert point_in_polygon((-1, -1), SQUARE) is False


def test_scale_polygon_identity_when_same_size():
    assert scale_polygon(SQUARE, (100, 100), (100, 100)) == SQUARE


def test_scale_polygon_scales_proportionally():
    scaled = scale_polygon(SQUARE, (10, 10), (20, 20))
    assert scaled[2] == (20, 20)


def test_scale_polygon_handles_zero_from_size():
    # frame_size (0,0) means "no zone config authored" — should be a no-op, not a crash
    assert scale_polygon(SQUARE, (0, 0), (100, 100)) == SQUARE


def test_euclidean_distance():
    assert euclidean((0, 0), (3, 4)) == 5.0
