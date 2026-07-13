import numpy as np
from data import canonical_orthogonal, covariance_family, paired_rotation

def test_spectrum_equality():
    family = covariance_family(20, 3, 75)
    spectra = [np.linalg.eigvalsh(x) for x in (family.target, family.auxiliary_1, family.auxiliary_2)]
    assert np.allclose(spectra[0], spectra[1]) and np.allclose(spectra[0], spectra[2])
    assert np.isclose(np.trace(family.target), 20) and np.isclose(np.linalg.cond(family.target), 9)

def test_covariance_spd():
    family = covariance_family(20, 3, 75)
    for covariance in (family.target, family.auxiliary_1, family.auxiliary_2):
        assert np.allclose(covariance, covariance.T)
        assert np.linalg.eigvalsh(covariance).min() > 0

def test_rotation_orthogonality():
    rotation = paired_rotation(12, 45)
    assert np.allclose(rotation.T @ rotation, np.eye(12))

def test_theta_zero_equality():
    family = covariance_family(10, 8, 0)
    assert np.allclose(family.target, family.auxiliary_1)
    assert np.allclose(family.target, family.auxiliary_2)

def test_target_invariant_across_theta():
    targets = [covariance_family(10, 4, theta).target for theta in (0, 45, 75)]
    assert np.array_equal(targets[0], targets[1]) and np.array_equal(targets[0], targets[2])
    assert np.array_equal(canonical_orthogonal(10, 4), canonical_orthogonal(10, 4))
