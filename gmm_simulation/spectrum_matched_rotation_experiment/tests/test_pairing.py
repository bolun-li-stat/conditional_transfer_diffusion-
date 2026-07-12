import numpy as np
from data import build_paired_split
from utils import setting_id

def test_reproducible_split():
    args = (8, 2, 45, 16, 16, 12, 12)
    first, second = build_paired_split(*args), build_paired_split(*args)
    assert all(np.array_equal(first[k], second[k]) for k in first)

def test_target_samples_shared_by_models():
    first = build_paired_split(8, 2, 45, 16, 16, 12, 12)
    assert np.array_equal(first["target_train"], first["joint_x"][:16])

def test_target_data_invariant_across_rotation():
    zero = build_paired_split(8, 2, 0, 16, 16, 12, 12)
    rotated = build_paired_split(8, 2, 75, 16, 16, 12, 12)
    for key in ("target_train", "target_val", "target_test"):
        assert np.array_equal(zero[key], rotated[key])

def test_target_only_setting_id_ignores_rotation():
    target_a = setting_id(0, "limited", 0, "target_only", 20, 8, 1.8, .2)
    target_b = setting_id(0, "limited", 75, "target_only", 20, 8, 1.8, .2)
    assert target_a == target_b

def test_joint_setting_id_contains_rotation():
    joint_a = setting_id(0, "limited", 0, "joint_conditional", 20, 8, 1.8, .2)
    joint_b = setting_id(0, "limited", 75, "joint_conditional", 20, 8, 1.8, .2)
    assert joint_a != joint_b
