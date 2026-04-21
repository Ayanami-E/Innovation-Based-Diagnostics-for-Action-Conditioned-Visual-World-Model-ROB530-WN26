"""Consistency diagnostics for Part 1."""

import numpy as np
from scipy import stats

from part1.se2 import wrap_angle


def nis_stats(nis_values, dim=3, alpha=0.05):
    """Summary statistics for a sample of NIS values vs. chi-squared(dim)."""
    nis = np.asarray(nis_values, dtype=float)
    nis = nis[np.isfinite(nis)]
    if nis.size == 0:
        return dict(mean_nis=np.nan, chi2_lo=np.nan, chi2_hi=np.nan,
                    frac_in_CI=np.nan, ks_stat=np.nan, ks_pvalue=np.nan,
                    n=0)
    chi2_lo = float(stats.chi2.ppf(alpha / 2, dim))
    chi2_hi = float(stats.chi2.ppf(1 - alpha / 2, dim))
    frac = float(np.mean((nis >= chi2_lo) & (nis <= chi2_hi)))
    ks = stats.kstest(nis, lambda x: stats.chi2.cdf(x, dim))
    return dict(
        mean_nis=float(np.mean(nis)),
        chi2_lo=chi2_lo,
        chi2_hi=chi2_hi,
        frac_in_CI=frac,
        ks_stat=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        n=int(nis.size),
    )


def rmse_pose(est, gt):
    """est, gt: (N, 3) arrays of [x, y, theta]. Returns (pos_rmse, ori_rmse)."""
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    pos_err = np.sqrt((est[:, 0] - gt[:, 0]) ** 2 +
                      (est[:, 1] - gt[:, 1]) ** 2)
    ori_err = np.array([wrap_angle(e - g)
                        for e, g in zip(est[:, 2], gt[:, 2])])
    return float(np.sqrt(np.mean(pos_err ** 2))), \
           float(np.sqrt(np.mean(ori_err ** 2)))
