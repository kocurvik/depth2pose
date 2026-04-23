from matplotlib import pyplot as plt

from depth_estimators.infer_depth import ALL_MDEs

def get_n_colors(n):
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i) for i in range(n)]
    return [plt.cm.hsv(i / n) for i in range(n)]

def get_mde_basename_color_dict():
    basenames = [x.split('Calib')[0] for x in sorted(ALL_MDEs.keys())]
    colors = get_n_colors(len(basenames))
    return {base_name: colors[i] for i, base_name in enumerate(basenames)}