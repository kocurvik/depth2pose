import pandas as pd
from prettytable import PrettyTable

from utils.results import get_mde_basename


def estimator_name(estimator):
    estimator_dict = {'calib': 'D', 'calib_shift': 'D$_{s}$',
                      'baseline_calib': 'B', 'baseline_sf': 'B$_{f}$',
                      'sf': 'D$_{f}$', 'sf_shift': 'D$_{s,f}$'}
    try:
        return estimator_dict[estimator]
    except KeyError:
        return estimator

def backbone_name(backbone):
    if 'vitl' in backbone or 'vit-l' in backbone or 'large' in backbone.lower():
        return 'ViT-L'
    if 'vits' in backbone or 'vit-s' in backbone or 'small' in backbone:
        return 'ViT-S'
    if 'vitb' in backbone or 'vit-b' in backbone or 'base' in backbone:
        return 'ViT-B'
    if 'giant' in backbone.lower():
        return 'ViT-G'
    if 'cvn' in backbone.lower():
        return 'ConvNext'
    if '' == backbone:
        return '-'
    return 'Default'



def print_tex_table(rows):
    print('\\begin{tabular}{cccccccc} \\')
    print('\multirow{2}{*}{Depth} & Inference & \multirow{2}{*}{Backbone} & \multirow{2}{*}{Estimator} & \\multicolumn{3}{c}{\\mAA} & MDE Runtime \\\\' )
    print('\\cline{5-7} \\\\')
    print('& with $\\M K$ &&& 10 iters & 100 iters & 1000 iters & (ms) \\\\ \\hline')

    for row in rows:
        basename = get_mde_basename(row[0])
        K_used = '\\checkmark' if 'Calib' in row[0] else ''
        backbone = backbone_name('-'.join(row[0].split('-')[1:]))
        estimator = estimator_name(row[1])
        mAAs_10 = row[2]['10']
        mAAs_100 = row[2]['100']
        mAAs_1000 = row[2]['1000']
        runtime = row[3]['mean_mde_runtime']
        new_row = [basename, K_used, backbone, estimator, mAAs_10, mAAs_100, mAAs_1000, runtime]
        str_row = [x if isinstance(x, str) else f'{x:0.2f}' for x in new_row]
        print('&'.join(str_row), '\\\\')

    print('\\end{tabular}')


def print_best_only_table(results_df, sort_rows=False, use_ro=False):
    calib_solvers = ['calib', 'calib_shift']
    uncal_solvers = ['mdecalib', 'mdecalib_shift', 'sf', 'sf_shift', 'vf', 'vf_shift']
    iter_values = [10, 100, 1000]

    if use_ro:
        calib_solvers = [f'{x}_ro' for x in calib_solvers]
        uncal_solvers = [f'{x}_ro' for x in uncal_solvers]

    # extra_metric_cols = [c for c in ['mean_mde_runtime', 'mean_inliers'] if c in results_df.columns]
    extra_metric_cols = [c for c in ['mean_mde_runtime'] if c in results_df.columns]
    iter_col_names = [str(i) for i in iter_values]

    def build_variant_dict(mde_list):
        variant_dict = {}
        for mde_name in mde_list:
            base_name = get_mde_basename(mde_name)
            if base_name not in variant_dict:
                variant_dict[base_name] = []
            variant_dict[base_name].append(mde_name)
        return variant_dict

    def get_iter_scores(df_all_iters, mde, solver):
        scores = {}
        for iters in iter_values:
            row = df_all_iters[(df_all_iters['mde'] == mde) & (df_all_iters['solver'] == solver) & (df_all_iters['iters'] == iters)]
            scores[str(iters)] = row['pose_mAA_10'].values[0] if not row.empty else None
        return scores

    def find_best(df_1000, df_all_iters, variants, solvers):
        sub = df_1000[df_1000['mde'].isin(variants) & df_1000['solver'].isin(solvers)]
        if sub.empty:
            return None
        best_row = sub.loc[sub['pose_mAA_10'].idxmax()]
        best_mde, best_solver = best_row['mde'], best_row['solver']
        iter_scores = get_iter_scores(df_all_iters, best_mde, best_solver)
        extra_vals = {c: best_row[c] for c in extra_metric_cols if c in best_row.index}
        return best_mde, best_solver, iter_scores, extra_vals

    def get_baseline_row(mde_label, solver_name, df_1000, df_all_iters):
        sub = df_1000[df_1000['solver'] == solver_name]
        if sub.empty:
            return None
        iter_scores = {}
        for iters in iter_values:
            r = df_all_iters[(df_all_iters['solver'] == solver_name) & (df_all_iters['iters'] == iters)]
            iter_scores[str(iters)] = r['pose_mAA_10'].mean() if not r.empty else None
        extra_vals = {c: sub[c].iloc[0] for c in extra_metric_cols if c in sub.columns}
        return mde_label, solver_name, iter_scores, extra_vals

    def print_table(rows, title):
        all_col_names = iter_col_names + extra_metric_cols
        tab = PrettyTable(['MDE', 'solver'] + all_col_names)
        tab.align['MDE'] = 'l'
        tab.align['solver'] = 'l'
        tab.float_format = '0.2'

        for row in rows:
            if row is None:
                continue
            mde, solver, iter_scores, extra_vals = row
            tab.add_row([mde, solver] + [iter_scores.get(c) for c in iter_col_names] + [extra_vals.get(c) for c in extra_metric_cols])
        print(f"**** {title} ****")
        print(tab)

    def collect_rows(df_1000, df_all_iters, sort_rows=False):
        variant_dict = build_variant_dict(df_1000['mde'].unique())
        calib_rows = []
        uncal_rows = []
        for base_mde, variants in variant_dict.items():
            calib_rows.append(find_best(df_1000, df_all_iters, variants, calib_solvers))
            uncal_variants = [v for v in variants if 'Calib' not in v]
            uncal_rows.append(find_best(df_1000, df_all_iters, uncal_variants, uncal_solvers))

        calib_rows.append(get_baseline_row('no_depth', 'baseline_calib', df_1000, df_all_iters))
        uncal_rows.append(get_baseline_row('no_depth', 'baseline_sf', df_1000, df_all_iters))
        uncal_rows.append(get_baseline_row('no_depth', 'baseline_vf', df_1000, df_all_iters))

        if sort_rows:
            calib_rows = sorted([r for r in calib_rows if r is not None],
                          key=lambda r: (r[2].get('1000') or 0), reverse=True)
            uncal_rows = sorted([r for r in uncal_rows if r is not None],
                          key=lambda r: (r[2].get('1000') or 0), reverse=True)

        return calib_rows, uncal_rows

    def make_grouped_dfs(subset_df):
        df_all_iters = subset_df.groupby(['mde', 'solver', 'iters'])['pose_mAA_10'].mean().reset_index()
        df_1000 = df_all_iters[df_all_iters['iters'] == 1000].copy()
        if extra_metric_cols:
            extra_df = subset_df[subset_df['iters'] == 1000].groupby(['mde', 'solver'])[extra_metric_cols].mean().reset_index()
            df_1000 = df_1000.merge(extra_df, on=['mde', 'solver'], how='left')
        return df_1000, df_all_iters

    # Per-dataset tables
    for dataset in sorted(results_df['dataset'].unique()):
        df_1000, df_all_iters = make_grouped_dfs(results_df[results_df['dataset'] == dataset])
        calib_rows, uncal_rows = collect_rows(df_1000, df_all_iters, sort_rows=sort_rows)
        print_table(calib_rows, f"{dataset} - Best Calib Results")
        print_table(uncal_rows, f"{dataset} - Best Uncal Results")

    # Mean over all datasets
    df_1000, df_all_iters = make_grouped_dfs(results_df)
    calib_rows, uncal_rows = collect_rows(df_1000, df_all_iters, sort_rows=sort_rows)
    print_table(calib_rows, "MEAN - Best Calib Results")
    print_table(uncal_rows, "MEAN - Best Uncal Results")
    print_tex_table(calib_rows)
    print_tex_table(uncal_rows)



if __name__ == '__main__':
    # results_df = pd.read_csv('csv_results/d2p_slim_pose_results.csv')
    results_df = pd.read_csv('csv_results/d2p_slim_pose_results.csv')
    print_best_only_table(results_df, sort_rows=True, use_ro=False)