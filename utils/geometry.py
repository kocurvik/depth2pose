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

    kp_x = np.clip(kp[:, 0].astype(int), 0, W - 1)
    kp_y = np.clip(kp[:, 1].astype(int), 0, H - 1)

    return depth_map[kp_y, kp_x]


    H, W = depth_map.shape
    # enforce this so linear interpolation with invalid values results in nan
    depth_map[depth_map <= 0] = np.inf
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
    depths = interp(kp[:, [1, 0]])
    return depths


def get_gt_depth(kp1, kp2, R_gt, t_gt, K1_gt, K2_gt):
    kp1_norm = np.linalg.inv(K1_gt) @ np.hstack((kp1, np.ones((kp1.shape[0], 1)))).T
    kp2_norm = np.linalg.inv(K2_gt) @ np.hstack((kp2, np.ones((kp2.shape[0], 1)))).T

    # Triangulate to get 3D points in camera 1 coordinate system
    # Using the standard triangulation formula for relative pose
    # x2 ~ R*x1 + t  =>  lambda2 * x2 = lambda1 * R * x1 + t
    # [R*x1  -x2] [lambda1; lambda2] = -t

    d1 = np.zeros(kp1.shape[0])
    d2 = np.zeros(kp1.shape[0])

    for i in range(kp1.shape[0]):
        A = np.zeros((3, 2))
        A[:, 0] = R_gt @ kp1_norm[:, i]
        A[:, 1] = -kp2_norm[:, i]
        b = -t_gt.flatten()

        # Solve least squares for depths (lambdas)
        depths, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        d1[i] = depths[0]
        d2[i] = depths[1]

    return d1, d2
