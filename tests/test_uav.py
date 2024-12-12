import numpy as np
from src.uav import UAV


def test_uav_initialization():
    map_pic = np.zeros((100, 100, 3), dtype=np.uint8)
    patch_size = 5

    uav = UAV(map_pic, patch_size)

    assert uav.map_pic.shape == (100, 100, 3)
    assert uav.patch_size == patch_size
    assert 0 <= uav.start_point < uav.height
    assert 0 <= uav.end_point < uav.height


def test_generate_trajectory():
    map_pic = np.zeros((100, 100, 3), dtype=np.uint8)
    patch_size = 5
    uav = UAV(map_pic, patch_size)
    seq_len = 10
    uav.generate_trajectory(seq_len)

    assert len(uav.traj_coords) == seq_len
    assert uav.localization == uav.traj_coords[0]


def test_set_patch():
    map_pic = np.zeros((100, 100, 3), dtype=np.uint8)
    patch_size = 5
    uav = UAV(map_pic, patch_size)
    seq_len = 10
    uav.generate_trajectory(seq_len)
    uav.set_patch()

    assert uav.patch.shape == (patch_size+1, patch_size+1, 3)


def test_move():
    map_pic = np.zeros((100, 100, 3), dtype=np.uint8)
    patch_size = 5
    uav = UAV(map_pic, patch_size)
    seq_len = 5
    uav.generate_trajectory(seq_len)

    initial_position = uav.get_position()
    finished = uav.move()
    assert finished is False
    assert uav.get_position() != initial_position

    for _ in range(seq_len-2):
        finished = uav.move()
    assert finished is True
