"""SE(2) Lie-group utilities.

Convention: xi = (rho_x, rho_y, phi) for se(2) 3-vectors
(translation-first, then rotation).
"""

import numpy as np

_J = np.array([[0.0, -1.0], [1.0, 0.0]])
_SMALL = 1e-9


def wrap_angle(a):
    """Wrap scalar angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def hat2(xi):
    xi = np.asarray(xi, dtype=float).reshape(3)
    rx, ry, phi = xi
    return np.array([[0.0, -phi, rx],
                     [phi,  0.0, ry],
                     [0.0,  0.0, 0.0]])


def vee2(M):
    M = np.asarray(M, dtype=float)
    phi = 0.5 * (M[1, 0] - M[0, 1])
    return np.array([M[0, 2], M[1, 2], phi])


def expm_se2(xi):
    xi = np.asarray(xi, dtype=float).reshape(3)
    rho = xi[:2]
    phi = xi[2]
    if abs(phi) < _SMALL:
        R = np.eye(2) + phi * _J
        V = np.eye(2) + 0.5 * phi * _J
    else:
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        V = np.array([[s / phi, -(1 - c) / phi],
                      [(1 - c) / phi, s / phi]])
    t = V @ rho
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def logm_se2(T):
    T = np.asarray(T, dtype=float)
    R = T[:2, :2]
    t = T[:2, 2]
    phi = np.arctan2(R[1, 0], R[0, 0])
    if abs(phi) < _SMALL:
        Vinv = np.eye(2) - 0.5 * phi * _J
    else:
        alpha = np.sin(phi) / phi
        beta = (1 - np.cos(phi)) / phi
        denom = alpha * alpha + beta * beta
        Vinv = np.array([[alpha, beta], [-beta, alpha]]) / denom
    rho = Vinv @ t
    return np.array([rho[0], rho[1], phi])


def pose_to_SE2(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])


def SE2_to_pose(T):
    T = np.asarray(T, dtype=float)
    theta = np.arctan2(T[1, 0], T[0, 0])
    return np.array([T[0, 2], T[1, 2], theta])


def adj_SE2(T):
    """Adjoint for (rho_x, rho_y, phi) ordering.

    Translation block is -J @ t with J = [[0,-1],[1,0]], i.e. [t_y, -t_x].
    """
    T = np.asarray(T, dtype=float)
    R = T[:2, :2]
    t = T[:2, 2]
    minusJt = np.array([t[1], -t[0]])
    Ad = np.zeros((3, 3))
    Ad[:2, :2] = R
    Ad[:2, 2] = minusJt
    Ad[2, 2] = 1.0
    return Ad


def invert_SE2(T):
    T = np.asarray(T, dtype=float)
    R = T[:2, :2]
    t = T[:2, 2]
    Ti = np.eye(3)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ t
    return Ti
