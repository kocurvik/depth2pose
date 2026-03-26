import numpy as np
from scipy.interpolate import RegularGridInterpolator


def R_err_fun(R1, R2):
    R_rel = np.dot(R1.T, R2)
    cos_theta = (np.trace(R_rel) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))


def t_err_fun(t1, t2):
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_theta = np.dot(t1, t2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))


def get_kp_depth(kp, depth_map, interpolation='linear'):
    H, W = depth_map.shape
    # Define the grid axes
    yy = np.arange(H)
    xx = np.arange(W)

    # Create the interpolator
    # 'linear' in Scipy is equivalent to bilinear for 2D grids
    interp = RegularGridInterpolator((yy, xx), depth_map,
                                     method=interpolation,
                                     bounds_error=False,
                                     fill_value=None)

    # Scipy expects (y, x) coordinates
    return interp(kp[:, [1, 0]])
