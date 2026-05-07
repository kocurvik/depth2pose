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

        higher_is_better = 'A.Rel' not in col and 'pose' not in col
        numeric_vals = pd.to_numeric(combined_df[col], errors='coerce')

        rank_mask = combined_df['mde'] != 'gt'
        max_rank = rank_mask.sum()
        ranks = numeric_vals[rank_mask].rank(ascending=not higher_is_better, method='min')
        ranks = ranks.reindex(combined_df.index)

        max_digits = 0
        for val in numeric_vals:
            if pd.isna(val):
                continue
            digits = len(str(int(abs(val))))
            max_digits = max(max_digits, digits)

        def format_cell(val, rank, max_d, max_r):
            if pd.isna(val) or val < 0.0:
                return ' '

            res = f"{val:.2f}"
            digits = len(str(int(abs(val))))
            prefix = "\\phantom{1}" * (max_d - digits)

            if pd.isna(rank):
                rank = 0
            else:
                rank = int(rank)

            res = f'\\rank{{{res}}}{{{rank}}}{{{max_r}}}'
            return prefix + res

        combined_df[col] = [format_cell(v, r, max_digits, max_rank) for v, r in zip(numeric_vals, ranks)]


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
    cline_range = f'{num_cols - solvers_count + 1}-{num_cols}'
    print(' & '.join(header) + f'\\\\  \\cmidrule{{{cline_range}}}')

    print(' & '.join(second_row) + '\\\\ \\midrule')

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


def get_solvers_depths(cal_type, depth_type, include_ro):
    if depth_type == 'scale':
        depth_cols = ['d1_si', 'A.Rel_si']
    elif depth_type == 'affine':
        depth_cols = ['d1_ssi', 'A.Rel_ssi']
    else:
        depth_cols = ['d1_si', 'A.Rel_si', 'd1_ssi', 'A.Rel_ssi']
    if cal_type == 'calib':
        solvers = ['calib']
        baseline = 'baseline_calib'
    else:
        solvers = ['sf', 'mdecalib']
        baseline = 'baseline_sf'
    if depth_type == 'affine':
        solvers = [f'{x}_shift' for x in solvers]
    elif depth_type == 'both':
        solvers += [f'{x}_shift' for x in solvers]
    if include_ro:
        solvers = [item for x in solvers for item in (x, f'{x}_ro')]
    return baseline, depth_cols, solvers


def print_combined_pivot_latex_table(depth_pivot, pose_pivot, depth_cols, solvers, groups,
                                     baselines=None, include_calib_col=True):
    def format_cell(val, rank, max_d, max_rank):
        if pd.isna(val) or val < 0.0:
            return ' '
        res = f"{val:.2f}"
        prefix = "\\phantom{1}" * (max_d - len(str(int(abs(val)))))
        if pd.isna(rank):
            rank = 0
        else:
            rank = int(rank)
        res = f'\\rank{{{res}}}{{{rank}}}{{{max_rank}}}'
        return prefix + res

    def format_column(series, higher_is_better):
        vals = pd.to_numeric(series, errors='coerce')
        rank_mask = series.index != 'gt'
        max_rank = int(rank_mask.sum())
        ranks = vals[rank_mask].rank(ascending=not higher_is_better, method='min').reindex(series.index)
        max_digits = max((len(str(int(abs(v)))) for v in vals if not pd.isna(v) and v >= 0), default=1)
        return [format_cell(v, r, max_digits, max_rank) for v, r in zip(vals, ranks)]

    formatted_depth = pd.DataFrame(index=depth_pivot.index, columns=depth_pivot.columns).astype(object)
    for col in depth_pivot.columns:
        formatted_depth[col] = format_column(depth_pivot[col], higher_is_better='A.Rel' not in col[1])

    formatted_pose = pd.DataFrame(index=pose_pivot.index, columns=pose_pivot.columns).astype(object)
    for col in pose_pivot.columns:
        formatted_pose[col] = format_column(pose_pivot[col], higher_is_better=True)

    n_depth = len(depth_cols)
    n_sol = len(solvers)
    n_per_group = n_depth + n_sol

    extra_cols = 1 + (1 if include_calib_col else 0)
    num_cols = extra_cols + len(groups) * n_per_group

    alignment = 'l' + ('c' * (num_cols - 1))
    print('\\begin{tabular}{' + alignment + '}')
    print('\\toprule')

    row1 = [' '] * extra_cols
    group_cmidrules = []
    cur = extra_cols + 1
    for group in groups:
        end = cur + n_per_group - 1
        row1.append('\\multicolumn{' + str(n_per_group) + '}{c}{' + str(group).capitalize() + '}')
        group_cmidrules.append(f'\\cmidrule(lr){{{cur}-{end}}}')
        cur = end + 1
    cmid_row1 = ' '.join(group_cmidrules)

    row2 = ['MDE-Backbone']
    if include_calib_col:
        row2.append('w/$\\M K$')
    for _ in groups:
        for col in depth_cols:
            row2.append(DEPTH_METRIC_MAP.get(col, col))
        for solver in solvers:
            row2.append(estimator_name(solver))

    print(' & '.join(row1) + ' \\\\ ' + cmid_row1)
    print(' & '.join(row2) + ' \\\\ \\midrule')

    for mde in depth_pivot.index:
        if mde == 'gt':
            mde_name = '\\midrule GT'
        else:
            basename = get_mde_basename(mde)
            backbone = get_backbone_name(mde, short=True)
            mde_name = f'{basename}-{backbone}'

        cells = [mde_name]
        if include_calib_col:
            cells.append('\\checkmark' if 'Calib' in mde else '')
        for group in groups:
            for col in depth_cols:
                cells.append(str(formatted_depth.loc[mde, (group, col)]))
            for solver in solvers:
                if mde in formatted_pose.index:
                    cells.append(str(formatted_pose.loc[mde, (group, solver)]))
                else:
                    cells.append(' ')
        print(' & '.join(cells) + ' \\\\')

    if baselines is not None:
        baseline_name = baselines.get('name', '')
        cells = [f'\\midrule No Depth + {estimator_name(baseline_name)}']
        if include_calib_col:
            cells.append('')
        for group in groups:
            for _ in depth_cols:
                cells.append(' ')
            val = baselines.get(group)
            inner = f'{val:.2f}' if val is not None else ' '
            cells.append('\\multicolumn{' + str(n_sol) + '}{c}{' + inner + '}')
        print(' & '.join(cells) + ' \\\\')

    print('\\bottomrule')
    print('\\end{tabular}')


def print_combined_table(results_df, depth_df, cal_type='calib', depth_type='scale', iters=1000, include_ro=True,
                         keep_only=None, best_only=False, dataset=None):
    baseline, depth_cols, solvers = get_solvers_depths(cal_type, depth_type, include_ro)

    if isinstance(dataset, (list, tuple)):
        depth_cols = ['d1_si']
        solvers = ['calib']
        datasets = list(dataset)
        results_df = results_df[results_df['dataset'].isin(datasets)]
        depth_df = depth_df[depth_df['dataset'].isin(datasets)]

        results_df = results_df.copy()[results_df['iters'] == iters]

        if cal_type == 'uncal':
            depth_df = depth_df[~depth_df['mde'].str.contains('Calib')]
            results_df = results_df[~results_df['mde'].str.contains('Calib')]

        depth_long = depth_df.groupby(['mde', 'dataset'])[depth_cols].mean().reset_index()
        depth_pivot = depth_long.pivot(index='mde', columns='dataset', values=depth_cols)
        depth_pivot = depth_pivot.swaplevel(axis=1).sort_index(axis=1)
        depth_pivot = depth_pivot.reindex(columns=pd.MultiIndex.from_product([datasets, depth_cols]))

        gt_row = pd.DataFrame([[-1.0] * len(depth_pivot.columns)],
                              index=['gt'], columns=depth_pivot.columns)
        depth_pivot = pd.concat([depth_pivot, gt_row])

        baselines = {'name': baseline}
        for d in datasets:
            b_df = results_df[(results_df['solver'] == baseline) & (results_df['dataset'] == d)]
            baselines[d] = b_df['pose_mAA_10'].mean() if not b_df.empty else None

        grouped = results_df.groupby(['mde', 'solver', 'dataset'])['pose_mAA_10'].mean().reset_index()
        pose_pivot = grouped.pivot(index='mde', columns=['dataset', 'solver'], values='pose_mAA_10')
        pose_pivot = pose_pivot.reindex(columns=pd.MultiIndex.from_product([datasets, solvers]))

        sort_metric = 'd1_ssi' if depth_type == 'affine' else 'd1_si'
        sort_series = depth_pivot.xs(sort_metric, axis=1, level=1).mean(axis=1)

        has_gt = 'gt' in depth_pivot.index
        non_gt = depth_pivot.drop(index='gt') if has_gt else depth_pivot
        sort_non_gt = sort_series.drop('gt') if has_gt else sort_series

        if best_only:
            basenames = non_gt.index.to_series().apply(get_mde_basename)
            sort_df = pd.DataFrame({'sort': sort_non_gt, 'basename': basenames})
            idx = sort_df.groupby('basename')['sort'].idxmax()
            non_gt = non_gt.loc[idx]
            sort_non_gt = sort_non_gt.loc[idx]

        order = sort_non_gt.sort_values(ascending=False).index
        non_gt = non_gt.loc[order]
        if keep_only is not None:
            non_gt = non_gt.head(keep_only)

        depth_pivot = pd.concat([non_gt, depth_pivot.loc[['gt']]]) if has_gt else non_gt
        pose_pivot = pose_pivot.reindex(depth_pivot.index)

        print_combined_pivot_latex_table(depth_pivot, pose_pivot, depth_cols, solvers, datasets,
                                         baselines=baselines, include_calib_col=cal_type == 'calib')
        return

    if dataset is not None:
        results_df = results_df[results_df['dataset'] == dataset]
        depth_df = depth_df[depth_df['dataset'] == dataset]

    depth_df = depth_df.groupby('mde')[depth_cols].mean().reset_index().copy()

    new_row = {col: -1.0 for col in depth_cols}
    new_row['mde'] = 'gt'
    depth_df = pd.concat([depth_df, pd.DataFrame([new_row])], ignore_index=True)

    results_df = results_df.copy()[results_df['iters'] == iters]

    if cal_type == 'uncal':
        depth_df = depth_df[~depth_df['mde'].str.contains('Calib')]
        results_df = results_df[~results_df['mde'].str.contains('Calib')]

    results_df = results_df.groupby(['solver', 'mde'])['pose_mAA_10'].mean().reset_index()
    baseline_pose_mAA = results_df[results_df['solver'] == baseline]['pose_mAA_10'].values[0]
    baseline = (baseline, baseline_pose_mAA)

    pivot_df = results_df.pivot(index='mde', columns='solver', values='pose_mAA_10')[solvers]
    combined_df = depth_df[['mde'] + depth_cols].merge(pivot_df, on='mde', how='left')
    combined_df = combined_df.fillna(-1.0)


    sort_metric = 'd1_ssi' if depth_type=='affine' else 'd1_si'

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


def print_pose_latex_table(pivot, baselines=None, include_calib_col=True):
    # pivot: index=mde, columns=MultiIndex(group, solver)
    groups = list(dict.fromkeys(pivot.columns.get_level_values(0)))
    solvers = list(dict.fromkeys(pivot.columns.get_level_values(1)))
    solvers_per_group = len(solvers)
    extra_cols = 1 + (1 if include_calib_col else 0)
    num_cols = extra_cols + len(groups) * solvers_per_group

    def format_cell(val, rank, max_d, max_rank):
        if pd.isna(val) or val < 0.0:
            return ' '
        res = f"{val:.2f}"
        prefix = "\\phantom{1}" * (max_d - len(str(int(abs(val)))))
        if pd.isna(rank):
            rank = 0
        else:
            rank = int(rank)
        res = f'\\rank{{{res}}}{{{rank}}}{{{max_rank}}}'
        return prefix + res

    formatted = pivot.copy().astype(object)
    for col in pivot.columns:
        vals = pd.to_numeric(pivot[col], errors='coerce')
        rank_mask = pivot.index != 'gt'
        max_rank = int(rank_mask.sum())
        ranks = vals[rank_mask].rank(ascending=False, method='min').reindex(pivot.index)
        max_digits = max((len(str(int(abs(v)))) for v in vals if not pd.isna(v) and v >= 0), default=1)
        formatted[col] = [format_cell(v, r, max_digits, max_rank) for v, r in zip(vals, ranks)]

    alignment = 'l' + ('c' * (num_cols - 1))
    print('\\begin{tabular}{' + alignment + '}')
    print('\\toprule')

    total_solver_cols = len(groups) * solvers_per_group
    maa_start = extra_cols + 1

    row1 = [' '] * extra_cols
    row1.append('\\multicolumn{' + str(total_solver_cols) + '}{c}{\\mAA}')
    maa_cmidrule = f'\\cmidrule(lr){{{maa_start}-{num_cols}}}'

    group_cmidrules = []
    row2 = [' ']
    if include_calib_col:
        row2.append('MDE')
    for i, group in enumerate(groups):
        row2.append('\\multicolumn{' + str(solvers_per_group) + '}{c}{' + group.capitalize() + '}')
        start = maa_start + i * solvers_per_group
        end = start + solvers_per_group - 1
        group_cmidrules.append(f'\\cmidrule(lr){{{start}-{end}}}')

    row3 = ['MDE-Backbone']
    if include_calib_col:
        row3.append('w/$\\M K$')
    for group in groups:
        for solver in solvers:
            row3.append(estimator_name(solver))

    print(' & '.join(row1) + ' \\\\ ' + maa_cmidrule)
    print(' & '.join(row2) + ' \\\\ ' + ' '.join(group_cmidrules))
    print(' & '.join(row3) + '\\\\ \\midrule')

    for mde, row in formatted.iterrows():
        if mde == 'gt':
            mde_name = '\\midrule GT'
        else:
            basename = get_mde_basename(mde)
            backbone = get_backbone_name(mde, short=True)
            mde_name = f'{basename}-{backbone}'

        cells = [mde_name]
        if include_calib_col:
            cells.append('\\checkmark' if 'Calib' in mde else '')
        for group in groups:
            for solver in solvers:
                cells.append(str(row[(group, solver)]))
        print(' & '.join(cells) + ' \\\\')

    if baselines is not None:
        cells = [f'\\midrule No Depth + {estimator_name(baselines["name"])}']
        if include_calib_col:
            cells.append('')
        for group in groups:
            val = baselines.get(group)
            inner = f'{val:.2f}' if val is not None else ' '
            cells.append('\\multicolumn{' + str(solvers_per_group) + '}{c}{' + inner + '}')
        print(' & '.join(cells) + ' \\\\')

    print('\\bottomrule')
    print('\\end{tabular}')


def print_join_latex_table(depth_df, standard_pivot, d2p_pivot, depth_cols, solvers, groups,
                           standard_baseline=None, d2p_baselines=None, include_calib_col=True):
    def format_cell(val, rank, max_d, max_rank):
        if pd.isna(val) or val < 0.0:
            return ' '
        res = f"{val:.2f}"
        prefix = "\\phantom{1}" * (max_d - len(str(int(abs(val)))))
        # if rank == 1:
        #     res = f"\\best{{{res}}}"
        # elif rank == 2:
        #     res = f"\\second{{{res}}}"
        # elif rank == 3:
        #     res = f"\\third{{{res}}}"
        if pd.isna(rank):
            rank = 0
        else:
            rank = int(rank)

        res = f'\\rank{{{res}}}{{{rank}}}{{{max_rank}}}'
        return prefix + res

    def format_column(series, higher_is_better):
        vals = pd.to_numeric(series, errors='coerce')
        rank_mask = series.index != 'gt'
        max_rank = rank_mask.sum()
        ranks = vals[rank_mask].rank(ascending=not higher_is_better, method='min').reindex(series.index)
        max_digits = max((len(str(int(abs(v)))) for v in vals if not pd.isna(v) and v >= 0), default=1)
        return [format_cell(v, r, max_digits, max_rank) for v, r in zip(vals, ranks)]

    formatted_depth = pd.DataFrame(index=depth_df.index)
    for col in depth_cols:
        formatted_depth[col] = format_column(depth_df[col], higher_is_better='A.Rel' not in col)

    formatted_std = pd.DataFrame(index=standard_pivot.index)
    for col in solvers:
        formatted_std[col] = format_column(standard_pivot[col], higher_is_better=True)

    formatted_d2p = pd.DataFrame(index=d2p_pivot.index, columns=d2p_pivot.columns).astype(object)
    for col in d2p_pivot.columns:
        formatted_d2p[col] = format_column(d2p_pivot[col], higher_is_better=True)

    n_depth = len(depth_cols)
    n_std = len(solvers)
    n_groups = len(groups)
    n_sol = len(solvers)
    n_d2p = n_groups * n_sol

    extra_cols = 1 + (1 if include_calib_col else 0)
    num_cols = extra_cols + n_depth + n_std + n_d2p

    depth_end = extra_cols + n_depth
    std_start = depth_end + 1
    std_end = std_start + n_std - 1
    d2p_start = std_end + 1

    alignment = 'l' + ('c' * (num_cols - 1))
    print('\\begin{tabular}{' + alignment + '}')
    print('\\toprule')

    row1 = [' '] * (extra_cols + n_depth)
    row1.append('\\multicolumn{' + str(n_std + n_d2p) + '}{c}{\\mAA}')
    cmid_row1 = f'\\cmidrule(lr){{{std_start}-{num_cols}}}'

    row2 = [' ']
    if include_calib_col:
        row2.append('MDE')
    for col in depth_cols:
        name = DEPTH_METRIC_MAP.get(col, col)
        row2.append('\\multirow{2}{*}{' + name + '}')
    row2.append('\\multicolumn{' + str(n_std) + '}{c}{Standard}')
    group_cmidrules = [f'\\cmidrule(lr){{{std_start}-{std_end}}}']
    cur = d2p_start
    for group in groups:
        end = cur + n_sol - 1
        row2.append('\\multicolumn{' + str(n_sol) + '}{c}{' + group.capitalize() + '}')
        group_cmidrules.append(f'\\cmidrule(lr){{{cur}-{end}}}')
        cur = end + 1
    cmid_row2 = ' '.join(group_cmidrules)

    row3 = ['MDE-Backbone']
    if include_calib_col:
        row3.append('w/$\\M K$')
    for _ in depth_cols:
        row3.append(' ')
    for solver in solvers:
        row3.append(estimator_name(solver))
    for _ in groups:
        for solver in solvers:
            row3.append(estimator_name(solver))

    print(' & '.join(row1) + ' \\\\ ' + cmid_row1)
    print(' & '.join(row2) + ' \\\\ ' + cmid_row2)
    print(' & '.join(row3) + ' \\\\ \\midrule')

    for mde in depth_df.index:
        if mde == 'gt':
            mde_name = '\\midrule GT'
        else:
            basename = get_mde_basename(mde)
            backbone = get_backbone_name(mde, short=True)
            mde_name = f'{basename}-{backbone}'

        cells = [mde_name]
        if include_calib_col:
            cells.append('\\checkmark' if 'Calib' in mde else '')
        for col in depth_cols:
            cells.append(str(formatted_depth.loc[mde, col]))
        for solver in solvers:
            if mde in formatted_std.index:
                cells.append(str(formatted_std.loc[mde, solver]))
            else:
                cells.append(' ')
        for group in groups:
            for solver in solvers:
                if mde in formatted_d2p.index:
                    cells.append(str(formatted_d2p.loc[mde, (group, solver)]))
                else:
                    cells.append(' ')
        print(' & '.join(cells) + ' \\\\')

    if standard_baseline is not None or d2p_baselines is not None:
        baseline_name = (d2p_baselines or {}).get('name', '')
        cells = [f'\\midrule No Depth + {estimator_name(baseline_name)}']
        if include_calib_col:
            cells.append('')
        for _ in depth_cols:
            cells.append(' ')
        std_val_str = f'{standard_baseline:.2f}' if standard_baseline is not None else ' '
        cells.append('\\multicolumn{' + str(n_std) + '}{c}{' + std_val_str + '}')
        for group in groups:
            val = d2p_baselines.get(group) if d2p_baselines else None
            inner = f'{val:.2f}' if val is not None else ' '
            cells.append('\\multicolumn{' + str(n_sol) + '}{c}{' + inner + '}')
        print(' & '.join(cells) + ' \\\\')

    print('\\bottomrule')
    print('\\end{tabular}')


def print_join_table(standard_results_df, d2p_results_df, depth_df, cal_type='calib',
                     depth_type='both', iters=1000, include_ro=True, keep_only=None, best_only=True):
    baseline_name, depth_cols, solvers = get_solvers_depths(cal_type, depth_type, include_ro)

    sort_metric = 'd1_si'
    agg_cols = list(depth_cols) + ([sort_metric] if sort_metric not in depth_cols else [])
    depth_df = depth_df.groupby('mde')[agg_cols].mean().reset_index().copy()
    new_row = {col: -1.0 for col in agg_cols}
    new_row['mde'] = 'gt'
    depth_df = pd.concat([depth_df, pd.DataFrame([new_row])], ignore_index=True)

    standard_results_df = standard_results_df.copy()[standard_results_df['iters'] == iters]
    d2p_results_df = d2p_results_df.copy()[d2p_results_df['iters'] == iters]
    d2p_results_df = d2p_results_df[~(d2p_results_df['group'] == 'ambiguos')]

    if cal_type == 'uncal':
        depth_df = depth_df[~depth_df['mde'].str.contains('Calib')]
        standard_results_df = standard_results_df[~standard_results_df['mde'].str.contains('Calib')]
        d2p_results_df = d2p_results_df[~d2p_results_df['mde'].str.contains('Calib')]

    standard_grouped = standard_results_df.groupby(['solver', 'mde'])['pose_mAA_10'].mean().reset_index()
    standard_baseline_vals = standard_grouped[standard_grouped['solver'] == baseline_name]['pose_mAA_10'].values
    standard_baseline_value = standard_baseline_vals[0] if len(standard_baseline_vals) > 0 else None
    standard_pivot = standard_grouped.pivot(index='mde', columns='solver', values='pose_mAA_10').reindex(columns=solvers)

    groups = ['mean', 'statues', 'vegetation']
    d2p_baselines = {'name': baseline_name}
    for group in groups:
        if group == 'mean':
            b_df = d2p_results_df[d2p_results_df['solver'] == baseline_name]
        else:
            b_df = d2p_results_df[(d2p_results_df['solver'] == baseline_name) & (d2p_results_df['group'] == group)]
        d2p_baselines[group] = b_df['pose_mAA_10'].mean() if not b_df.empty else None

    group_results_df = d2p_results_df.groupby(['solver', 'mde', 'group'])['pose_mAA_10'].mean().reset_index()
    mean_results_df = d2p_results_df.groupby(['solver', 'mde'])['pose_mAA_10'].mean().reset_index()
    mean_results_df['group'] = 'mean'
    d2p_all = pd.concat([group_results_df, mean_results_df], axis=0)
    d2p_pivot = d2p_all[d2p_all['group'].isin(groups)].pivot(
        index='mde', columns=['group', 'solver'], values='pose_mAA_10')
    d2p_pivot = d2p_pivot.reindex(columns=pd.MultiIndex.from_product([groups, solvers]))

    combined = depth_df.set_index('mde')

    if best_only:
        combined['_basename'] = combined.index.to_series().apply(get_mde_basename)
        idx = combined.groupby('_basename')[sort_metric].idxmax()
        combined = combined.loc[idx].drop(columns=['_basename'])

    combined = combined.sort_values(by=sort_metric, ascending=False)
    if keep_only is not None:
        combined = combined.head(keep_only)

    if 'gt' in combined.index:
        combined = pd.concat([combined.drop(index='gt'), combined.loc[['gt']]])

    selected_mdes = list(combined.index)
    standard_pivot = standard_pivot.reindex(selected_mdes)
    d2p_pivot = d2p_pivot.reindex(selected_mdes)
    depth_df_selected = combined[depth_cols]

    print_join_latex_table(depth_df_selected, standard_pivot, d2p_pivot, depth_cols, solvers, groups,
                           standard_baseline=standard_baseline_value,
                           d2p_baselines=d2p_baselines,
                           include_calib_col=cal_type == 'calib')


def print_pose_table(results_df, cal_type='calib', depth_type='scale', iters=1000, include_ro=True, best_only=True):
    baseline_name, depth_cols, solvers = get_solvers_depths(cal_type, depth_type, include_ro)

    solvers = [x for x in solvers if 'mde' not in x]

    results_df = results_df.copy()[results_df['iters'] == iters]
    results_df = results_df[~(results_df['group'] == 'ambiguos')]

    if cal_type == 'uncal':
        results_df = results_df[~results_df['mde'].str.contains('Calib')]

    groups = ['mean', 'statues', 'vegetation']

    baselines = {'name': baseline_name}
    for group in groups:
        if group == 'mean':
            b_df = results_df[results_df['solver'] == baseline_name]
        else:
            b_df = results_df[(results_df['solver'] == baseline_name) & (results_df['group'] == group)]
        baselines[group] = b_df['pose_mAA_10'].mean() if not b_df.empty else None

    group_results_df = results_df.groupby(['solver', 'mde', 'group'])['pose_mAA_10'].mean().reset_index()
    mean_results_df = results_df.groupby(['solver', 'mde'])['pose_mAA_10'].mean().reset_index()
    mean_results_df['group'] = 'mean'

    all_results_df = pd.concat([group_results_df, mean_results_df], axis=0)

    if best_only:
        mean_first = all_results_df[(all_results_df['group'] == 'mean') &
                                    (all_results_df['solver'] == solvers[0])].copy()
        mean_first['basename'] = mean_first['mde'].apply(get_mde_basename)
        idx = mean_first.groupby('basename')['pose_mAA_10'].idxmax()
        keep_mdes = mean_first.loc[idx, 'mde'].values
        all_results_df = all_results_df[all_results_df['mde'].isin(keep_mdes)]

    pivot = all_results_df[all_results_df['group'].isin(groups)].pivot(
        index='mde', columns=['group', 'solver'], values='pose_mAA_10')
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([groups, solvers]))
    pivot = pivot.dropna(how='all')
    pivot = pivot.sort_values(by=('mean', solvers[0]), ascending=False)
    if 'gt' in pivot.index:
        pivot = pd.concat([pivot.drop(index='gt'), pivot.loc[['gt']]])

    print_pose_latex_table(pivot, baselines=baselines, include_calib_col=cal_type == 'calib')





if __name__ == '__main__':
    depth_df = pd.read_csv('csv_results/standard_depth_results.csv')

    matches_list = ['loma', 'splg']

    for matches in matches_list:
        d2p_results = pd.read_csv(f'csv_results/d2p_{matches}_slim_pose_results.csv')
        standard_results = pd.read_csv(f'csv_results/standard_{matches}_slim_pose_results.csv')

        # print("-------MAIN PAPER TABLE ------")
        # print()
        # print_join_table(standard_results, d2p_results, depth_df, cal_type='calib', depth_type='scale', include_ro=True, keep_only=None, best_only=True)


        # appendix
        # print("-------- D2P Appendix Table ---------")
        # print_pose_table(d2p_results, cal_type='calib', depth_type='both', include_ro=True, best_only=False)

        # main paper
        # appendix
        # print("-------APPENDIX TABLE ------")
        # print_combined_table(standard_results, depth_df, cal_type='calib', depth_type='both', include_ro=True)
        # print("-------APPENDIX TABLE ------")
        # print_combined_table(standard_results, depth_df, cal_type='uncal', depth_type='both', include_ro=True)
        print("-------APPENDIX TABLE ------")
        print_combined_table(standard_results, depth_df, cal_type='calib', depth_type='scale', include_ro=False, dataset=['eth3d', 'scannetpp', 'lamar', 'sintel'])




