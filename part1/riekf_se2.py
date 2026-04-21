"""Right-Invariant EKF on SE(2) with body-frame twist velocity.

Reference:
    Barrau, A., & Bonnabel, S. (2017). "The Invariant Extended Kalman Filter
    as a Stable Observer." IEEE Trans. Automatic Control, 62(4), 1797-1812.

State:
    X in SE(2)  -- 3x3 homogeneous pose
    v in R^3    -- body-frame twist (v_bx, v_by, omega)

Error state (6-D) ordering: [xi_world (3); dv_body (3)].

Prediction:
    X_{t+1} = X_t @ exp((v + w_xi) * dt)
    v_{t+1} = v + w_v

Linearized about (X_hat, v):
    F = [[ I_3,   Ad(X_hat) * dt ],
         [ 0_3,   I_3            ]]
    Q_eff = blkdiag(Q_xi * dt, Q_v * dt)

Measurement (world-frame pose Y):
    eta = Y @ inv(X_hat)
    r   = log(eta)                      # 3-vector, world-frame error
    H   = [ I_3, 0_3 ]
    S   = H P H^T + R
    K   = P H^T S^{-1}
    X  <- exp(K[0:3] @ r) @ X           # left-multiplication retraction
    v  <- v + K[3:6] @ r
    P  <- (I - K H) P
"""

import numpy as np

from part1.se2 import (
    adj_SE2,
    expm_se2,
    invert_SE2,
    logm_se2,
    pose_to_SE2,
    SE2_to_pose,
)


class RIEKF_SE2:
    def __init__(self, dt, Q_xi, Q_v, R_meas, X0, v0, P0):
        self.dt = float(dt)
        self.Q_xi = np.asarray(Q_xi, dtype=float).copy()
        self.Q_v = np.asarray(Q_v, dtype=float).copy()
        self.R = np.asarray(R_meas, dtype=float).copy()
        assert self.Q_xi.shape == (3, 3)
        assert self.Q_v.shape == (3, 3)
        assert self.R.shape == (3, 3)

        self.X = np.asarray(X0, dtype=float).copy()
        assert self.X.shape == (3, 3)
        self.v = np.asarray(v0, dtype=float).reshape(3).copy()

        self.P = np.asarray(P0, dtype=float).copy()
        assert self.P.shape == (6, 6)

        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        self.nis_log = []
        self.innov_log = []

    def predict(self):
        dt = self.dt
        Ad = adj_SE2(self.X)

        # State propagation
        self.X = self.X @ expm_se2(self.v * dt)

        # Covariance propagation
        F = np.eye(6)
        F[0:3, 3:6] = Ad * dt
        Qeff = np.zeros((6, 6))
        Qeff[0:3, 0:3] = self.Q_xi * dt
        Qeff[3:6, 3:6] = self.Q_v * dt
        self.P = F @ self.P @ F.T + Qeff

    def update(self, y):
        """y: (3,) world-frame pose measurement [x, y, theta]."""
        y = np.asarray(y, dtype=float).reshape(3)
        Y = pose_to_SE2(y[0], y[1], y[2])

        eta = Y @ invert_SE2(self.X)
        r = logm_se2(eta)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        dxi = K[0:3, :] @ r
        dv = K[3:6, :] @ r

        self.X = expm_se2(dxi) @ self.X
        self.v = self.v + dv

        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        nis = float(r @ np.linalg.solve(S, r))
        self.nis_log.append(nis)
        self.innov_log.append(r.copy())

    def pose(self):
        return SE2_to_pose(self.X)
