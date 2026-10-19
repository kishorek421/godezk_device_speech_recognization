import numpy as np
import pytest

from walker_audio.spatial import camera_x_to_azimuth, delay_and_sum


def test_camera_position_maps_to_azimuth() -> None:
    assert camera_x_to_azimuth(0, 1000, 80) == pytest.approx(-40)
    assert camera_x_to_azimuth(500, 1000, 80) == pytest.approx(0)
    assert camera_x_to_azimuth(1000, 1000, 80) == pytest.approx(40)


def test_delay_and_sum_rejects_wrong_channel_count() -> None:
    positions = np.asarray([[-0.03, 0, 0], [0.03, 0, 0]])
    with pytest.raises(ValueError, match="configured microphones"):
        delay_and_sum(np.zeros((100, 1)), 16_000, positions, 0)
