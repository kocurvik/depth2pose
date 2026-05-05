from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from utils.results import get_mde_basename, get_backbone_name


def estimator_name(estimator):
    if estimator == 'sf':
        return '\\mysf{}'
    no_underscore = ''.join(estimator.split('_'))
    return f'\\{no_underscore}{{}}'

DEPTH_METRIC_MAP = {
    'd1_si': r'\dsi',
    'd1_ssi': r'\dssi',
    'A.Rel_si': r'\relsi',
    'A.Rel_ssi': r'\relssi'
}

def print_tex_table(rows):
    print('\\begin{tabular}{cccccccc} \\')
    print('\multirow{2}{*}{Depth} & Inference & \multirow{2}{*}{Backbone} & \multirow{2}{*}{Estimator} & \\multicolumn{3}{c}{\\mAA} & MDE Runtime \\\\' )
    print('\\cline{5-7} \\\\')
    print('& with $\\M K$ &&& 10 iters & 100 iters & 1000 iters & (ms) \\\\ \\hline')

    for row in rows:
        basename = get_mde_basename(row[0])
        K_used = '\\checkmark' if 'Calib' in row[0] else ''
        backbone = get_backbone_name('-'.join(row[0].split('-')[1:]))
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
    # calib_solvers = ['calib', 'calib_shift']
    calib_solvers = ['calib']
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

        calib_rows.append(get_baseline_row('none', 'baseline_calib', df_1000, df_all_iters))
        uncal_rows.append(get_baseline_row('none', 'baseline_sf', df_1000, df_all_iters))
        uncal_rows.append(get_baseline_row('none', 'baseline_vf', df_1000, df_all_iters))

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


def print_best_only_depth_table(depth_results_df, sort_rows=True):
    metric_cols = ['A.Rel_si', 'd1_si', 'A.Rel_ssi', 'd1_ssi']

    summary_df = depth_results_df.groupby(['mde'])[metric_cols].mean().reset_index()
    summary_df['basename'] = summary_df['mde'].apply(get_mde_basename)

    best_rows = summary_df.loc[summary_df.groupby('basename')['A.Rel_ssi'].idxmin()]
    best_rows = best_rows.drop(columns=['basename'])

    if sort_rows:
        best_rows = best_rows.sort_values(by='A.Rel_ssi', ascending=True)

    tab = PrettyTable(["MDE"] + metric_cols)

    tab.float_format = ".4"

    tab.add_rows(best_rows.values.tolist())

    print(tab)


def df_vals_to_latex_string(combined_df):
    for col in combined_df.columns:
        if col == 'mde':
            continue

        higher_is_better = 'A.Rel' not in col
        numeric_vals = pd.to_numeric(combined_df[col], errors='coerce')

        rank_mask = combined_df['mde'] != 'gt'
        ranks = numeric_vals[rank_mask].rank(ascending=not higher_is_better, method='min')
        ranks = ranks.reindex(combined_df.index)

        max_digits = 0
        for val in numeric_vals:
            if pd.isna(val):
                continue
            digits = len(str(int(abs(val))))
            max_digits = max(max_digits, digits)

        def format_cell(val, rank, max_d):
            if np.isnan(val) or val < 0.0:
                return ' '

            if pd.isna(val) or val < 0.0:
                return " "

            res = f"{val:.2f}"
            digits = len(str(int(abs(val))))
            prefix = "\\phantom{1}" * (max_d - digits)

            if rank == 1:
                res = f"\\best{{{res}}}"
            elif rank == 2:
                res =  f"\\second{{{res}}}"
            elif rank == 3:
                res = f"\\third{{{res}}}"

            res = prefix + res

            return res

        combined_df[col] = [format_cell(v, r, max_digits) for v, r in zip(numeric_vals, ranks)]


def print_combined_latex_table(combined_df, baseline=None, include_calib_col=True):
    num_cols = len(combined_df.columns)
    if include_calib_col:
        num_cols += 1
    # Adjust column alignment: l for name, rest c
    alignment = 'l' + ('c' * (num_cols - 1))
    print('\\begin{tabular}{' + alignment + '}')

    # Header
    header = ['\\multirow{2}{*}{MDE-Backbone}']
    if include_calib_col:
        header.append('MDE')

    second_row_metrics = []

    solvers_count = 0

    for col in combined_df.columns:
        if col == 'mde':
            continue
        # Use depth_metric_name or estimator_name
        if col in DEPTH_METRIC_MAP:
            name = DEPTH_METRIC_MAP[col]
            second_row_metrics.append(' ')
            header.append('\\multirow{2}{*}{' + name + '}')
        else:
            second_row_metrics.append(estimator_name(col))
            solvers_count += 1



    second_row = [' ']
    if include_calib_col:
        second_row += ['w/$\M K$']
    second_row += second_row_metrics

    header += ['\\multicolumn{' + str(solvers_count) + '}{c}{\\mAA}']
    print('\\toprule')
    print(' & '.join(header) + ' \\\\ ')
    print(' & '.join(second_row) + ' \\\\ \\midrule')

    for _, row in combined_df.iterrows():
        mde = row['mde']
        if mde == 'gt':
            mde_name = '\\midrule GT'
        else:
            basename = get_mde_basename(mde)
            backbone = get_backbone_name(mde, short=True)
            mde_name = f'{basename}-{backbone}'

        cells = [mde_name]
        if include_calib_col:
            calib_symbol = '\\checkmark' if 'Calib' in mde else ''
            cells.append(calib_symbol)

        for col in combined_df.columns:
            if col == 'mde':
                continue
            cells.append(str(row[col]))

        print(' & '.join(cells) + ' \\\\')

    if baseline is not None:
        val = f"{baseline[1]:.2f}"
        print(f'\\midrule No Depth + {estimator_name(baseline[0])}' + (num_cols - solvers_count) * '&' +
              '\\multicolumn{' + str(solvers_count) +'}{c}{'+val+'}\\\\')
    print('\\bottomrule')

    print('\\end{tabular}')




def print_combined_table(results_df, depth_df, cal_type='calib', depth_type='scale', iters=1000, include_ro=True,
                         keep_only=8, best_only=False):
    if depth_type == 'scale':
        depth_cols = ['d1_si', 'A.Rel_si']
    else:
        depth_cols = ['d1_ssi', 'A.Rel_ssi']

    if cal_type == 'calib':
        solvers = ['calib']
        baseline = 'baseline_calib'
    else:
        solvers = ['sf']
        baseline = 'basline_sf'


    if depth_type != 'scale':
        solvers = [f'{x}_shift' for x in solvers]

    if include_ro:
        solvers += [f'{x}_ro' for x in solvers]

    depth_df = depth_df.groupby('mde')[depth_cols].mean().reset_index().copy()

    new_row = {col: -1.0 for col in depth_cols}
    new_row['mde'] = 'gt'
    depth_df = pd.concat([depth_df, pd.DataFrame([new_row])], ignore_index=True)

    results_df = results_df.copy()[results_df['iters'] == iters]
    results_df = results_df.groupby(['solver', 'mde'])['pose_mAA_10'].mean().reset_index()
    baseline_pose_mAA = results_df[results_df['solver'] == baseline]['pose_mAA_10'][0]
    baseline = (baseline, baseline_pose_mAA)

    pivot_df = results_df.pivot(index='mde', columns='solver', values='pose_mAA_10')[solvers]
    combined_df = depth_df[['mde'] + depth_cols].merge(pivot_df, on='mde', how='left')

    sort_metric = 'd1_si' if depth_type=='scale' else 'd1_ssi'

    if best_only:
        combined_df['basename'] = combined_df['mde'].apply(get_mde_basename)
        idx = combined_df.groupby('basename')[sort_metric].idxmax()
        combined_df = combined_df.loc[idx].drop(columns=['basename'])


    combined_df = combined_df.dropna(axis=0)
    combined_df = combined_df.sort_values(by=sort_metric, ascending=False)
    if keep_only is not None:
        combined_df = combined_df.head(keep_only)

    df_vals_to_latex_string(combined_df)

    print_combined_latex_table(combined_df, baseline=baseline, include_calib_col=cal_type == 'calib')



if __name__ == '__main__':
    # results_df = pd.read_csv('csv_results/d2p_slim_pose_results.csv')
    results_df = pd.read_csv('csv_results/standard_splg_slim_pose_results.csv')
    depth_df = pd.read_csv('csv_results/standard_depth_results.csv')

    print_combined_table(results_df, depth_df, cal_type='calib', keep_only=None, include_ro=True, best_only=True)
    # print_combined_table(results_df, depth_df, cal_type='uncal', keep_only=None)
    # print_combined_table(results_df, depth_df, cal_type='calib', depth_type='affine', keep_only=8)
    # print_combined_table(results_df, depth_df, cal_type='uncal', depth_type='affine', keep_only=8)

    # print_best_only_table(results_df, sort_rows=True, use_ro=False)
    #
    # depth_results_df = pd.read_csv('csv_results/standard_depth_results.csv')
    # print_best_only_depth_table(depth_results_df, sort_rows=True)