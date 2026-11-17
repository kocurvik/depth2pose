"""Check that the installed PoseLib provides the monocular depth solvers used by the benchmark."""

POSELIB_URL = 'https://github.com/kocurvik/PoseLib/tree/mdrpbench'
POSELIB_INSTALL = 'pip install git+https://github.com/kocurvik/PoseLib.git@mdrpbench'
REQUIRED = ('Camera', 'MonoDepthImagePair', 'MonoDepthTwoViewGeometry', 'estimate_monodepth_relative_pose')


def _smoke_test(poselib):
    """Estimate a relative pose from noise-free synthetic data with the options used in eval_pose.py."""
    import numpy as np

    rng = np.random.default_rng(0)
    K = np.array([[1000.0, 0, 500], [0, 1000, 400], [0, 0, 1]])
    X = rng.uniform([-2, -2, 4], [2, 2, 8], (100, 3))
    angle = 0.1
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    t = np.array([-1.0, 0.1, 0.2])
    X2 = X @ R.T + t
    kp1 = (X / X[:, 2:]) @ K.T
    kp2 = (X2 / X2[:, 2:]) @ K.T
    d1, d2 = X[:, 2] * 3.0, X2[:, 2] * 3.0  # depth known only up to scale

    camera = poselib.Camera({'model': 'PINHOLE', 'width': -1, 'height': -1, 'params': [1000.0, 1000.0, 500.0, 400.0]})
    ransac = {'max_iterations': 100, 'min_iterations': 100, 'progressive_sampling': False}
    for weight_sampson in (1.0, -1.0):  # calib and calib_ro
        options = {'max_errors': [16.0, 2.0], 'estimate_shift': False, 'ransac': ransac}
        if weight_sampson < 0:
            options['weight_sampson'] = weight_sampson
        pose, info = poselib.estimate_monodepth_relative_pose(kp1[:, :2], kp2[:, :2], d1, d2, camera, camera, options)
        pair = poselib.MonoDepthImagePair(pose, camera, camera)
        R_est = pair.geometry.pose.R
        cos = np.clip((np.trace(R_est.T @ R) - 1) / 2, -1, 1)
        if np.degrees(np.arccos(cos)) > 0.1 or 'inliers' not in info:
            raise RuntimeError('the solver returned a wrong pose on noise-free synthetic data')


def check_poselib():
    """Returns None if PoseLib is usable, otherwise a description of the problem."""
    try:
        import poselib
    except ImportError as e:
        return f'PoseLib is not installed ({e}).'

    location = getattr(poselib, '__file__', '?')
    missing = [x for x in REQUIRED if not hasattr(poselib, x)]
    if missing:
        return f'The installed PoseLib ({location}) does not provide {", ".join(missing)}.'
    try:
        _smoke_test(poselib)
    except Exception as e:
        # pybind errors list all signatures over many lines, the first line is enough
        reason = (str(e).strip().splitlines() or [type(e).__name__])[0]
        reason = reason.split(' The following argument types are supported')[0]
        return f'The installed PoseLib ({location}) is not compatible with this benchmark: {reason}'
    return None


def warn_poselib():
    """Print a warning with installation instructions if PoseLib is not usable. Returns True if it is usable."""
    problem = check_poselib()
    if problem is None:
        return True
    print('=' * 100)
    print(f'WARNING: {problem}')
    print('The benchmark needs the monocular depth relative pose solvers (RePoseD) from our PoseLib fork:')
    print(f'    {POSELIB_URL}')
    print('Install it into this environment with:')
    print(f'    {POSELIB_INSTALL}')
    print('=' * 100)
    return False
