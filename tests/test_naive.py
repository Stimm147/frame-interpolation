import numpy as np
from interpolation_app.interpolation.naive import NaiveInterpolator


def test_naive_interpolation_shape_and_type():
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
    frame2 = np.ones((64, 64, 3), dtype=np.uint8) * 255

    interpolator = NaiveInterpolator()
    result = interpolator.interpolate(frame1, frame2)

    assert result.shape == frame1.shape
    assert result.dtype == np.uint8
    assert np.all(result == 127)
