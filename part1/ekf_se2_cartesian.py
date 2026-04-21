"""EKF-Cartesian baseline.

6-D state [x, y, theta, vx, vy, omega] with constant-velocity dynamics.
Linear F, H; theta innovation wrapped via atan2(sin, cos).
"""

import numpy as np

from part1.se2 import wrap_angle


class EKF_SE2_Cartesian:
    def __init__(self, dt, Q, R, x0, P0):
        self.dt = float(dt)
        self.Q = np.asarray(Q, dtype=float).copy()
        self.R = np.asarray(R, dtype=float).copy()
        assert self.Q.shape == (6, 6)
        assert self.R.shape == (3, 3)

        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.x = np.asarray(x0, dtype=float).reshape(6).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        assert self.P.shape == (6, 6)

        self.nis_log = []
        self.innov_log = []

    def predict(self):
        self.x = self.F @ self.x
        self.x[2] = wrap_angle(self.x[2])
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y):
        y = np.asarray(y, dtype=float).reshape(3)
        v = y - self.H @ self.x
        v[2] = wrap_angle(v[2])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ v
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        nis = float(v @ np.linalg.solve(S, v))
        self.nis_log.append(nis)
        self.innov_log.append(v.copy())

    def pose(self):
        return np.array([self.x[0], self.x[1], wrap_angle(self.x[2])])
