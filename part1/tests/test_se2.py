"""Lie-group sanity tests for part1.se2."""

import numpy as np
import pytest

from part1 import se2


def _random_xi(rng, max_phi=np.pi - 1e-3):
    return np.array([rng.uniform(-0.5, 0.5),
                     rng.uniform(-0.5, 0.5),
                     rng.uniform(-max_phi, max_phi)])


def _random_SE2(rng):
    return se2.expm_se2(_random_xi(rng))


def test_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        X = _random_SE2(rng)
        X2 = se2.expm_se2(se2.logm_se2(X))
        assert np.allclose(X, X2, atol=1e-10), (X, X2)


def test_log_exp_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        xi = _random_xi(rng)
        xi2 = se2.logm_se2(se2.expm_se2(xi))
        assert np.allclose(xi, xi2, atol=1e-10), (xi, xi2)


def test_adjoint_identity():
    rng = np.random.default_rng(2)
    for _ in range(100):
        X = _random_SE2(rng)
        Xinv = se2.invert_SE2(X)
        xi = _random_xi(rng)
        lhs = X @ se2.hat2(xi) @ Xinv
        rhs = se2.hat2(se2.adj_SE2(X) @ xi)
        assert np.allclose(lhs, rhs, atol=1e-10), (lhs, rhs)


def test_small_angle_branch():
    # Very small phi should not blow up.
    for phi in [0.0, 1e-12, -1e-12, 1e-10]:
        xi = np.array([0.1, -0.2, phi])
        X = se2.expm_se2(xi)
        xi_back = se2.logm_se2(X)
        assert np.allclose(xi, xi_back, atol=1e-9)


def test_invert_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        X = _random_SE2(rng)
        I = X @ se2.invert_SE2(X)
        assert np.allclose(I, np.eye(3), atol=1e-10)


def test_pose_se2_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(20):
        x, y = rng.uniform(-1, 1, 2)
        th = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
        T = se2.pose_to_SE2(x, y, th)
        p = se2.SE2_to_pose(T)
        assert np.allclose(p, [x, y, th], atol=1e-12)


def test_wrap_angle():
    assert abs(se2.wrap_angle(3 * np.pi) - np.pi) < 1e-12 or \
           abs(se2.wrap_angle(3 * np.pi) + np.pi) < 1e-12
    assert abs(se2.wrap_angle(-np.pi - 0.1) - (np.pi - 0.1)) < 1e-12
