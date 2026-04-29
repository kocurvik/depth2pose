import pandas as pd
from prettytable import PrettyTable


def print_best_only_table(results_df, sort_rows=False):
    calib_solvers = ['calib', 'calib_shift']
    uncal_solvers = ['mdecalib', 'mdecalib_shift', 'sf', 'sf_shift', 'vf', 'vf_shift']

    df = results_df[results_df['iters'] == 1000]

    metric_cols = [c for c in ['pose_mAA_10', 'mean_mde_runtime', 'mean_inliers'] if c in df.columns]

    def build_variant_dict(sub_df):
        variant_dict = {}
        for mde_name in sub_df['mde'].unique():
            base_name = mde_name.split('-')[0].split('Calib')[0]
            if base_name not in variant_dict:
                variant_dict[base_name] = []
            variant_dict[base_name].append(mde_name)
        return variant_dict

    def find_best(sub_df, variants, solvers):
        sub = sub_df[sub_df['mde'].isin(variants) & sub_df['solver'].isin(solvers)]
        if sub.empty:
            return None
        best_row = sub.loc[sub['pose_mAA_10'].idxmax()]
        return best_row['mde'], best_row['solver'], best_row[metric_cols]

    def print_table(rows, title):
        tab = PrettyTable(['MDE', 'solver'] + metric_cols)
        tab.align['MDE'] = 'l'
        tab.align['solver'] = 'l'
        tab.float_format = '0.2'
        if sort_rows:
            rows = sorted([r for r in rows if r is not None], key=lambda r: r[2]['pose_mAA_10'], reverse=True)
        for row in rows:
            if row is None:
                continue
            mde, solver, means = row
            tab.add_row([mde, solver] + [means[c] for c in metric_cols])
        print(f"**** {title} ****")
        print(tab)

    def collect_rows(sub_df):
        variant_dict = build_variant_dict(sub_df)
        calib_rows = []
        uncal_rows = []
        for base_mde, variants in variant_dict.items():
            if base_mde == 'gt':
                continue
            calib_rows.append(find_best(sub_df, variants, calib_solvers))
            uncal_variants = [v for v in variants if 'Calib' not in v]
            uncal_rows.append(find_best(sub_df, uncal_variants, uncal_solvers))

        gt_df = sub_df[sub_df['mde'] == 'gt']
        if not gt_df.empty:
            gt_calib = gt_df[gt_df['solver'] == 'baseline_calib']
            if not gt_calib.empty:
                calib_rows.append(('gt', 'baseline_calib', gt_calib[metric_cols].iloc[0]))
            for solver in ['baseline_sf', 'baseline_vf']:
                gt_rows = gt_df[gt_df['solver'] == solver]
                if not gt_rows.empty:
                    uncal_rows.append(('gt', solver, gt_rows[metric_cols].iloc[0]))
                    break

        return calib_rows, uncal_rows

    # Per-dataset tables
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset].groupby(['mde', 'solver'])[metric_cols].mean().reset_index()
        calib_rows, uncal_rows = collect_rows(dataset_df)
        print_table(calib_rows, f"{dataset} - Best Calib Results")
        print_table(uncal_rows, f"{dataset} - Best Uncal Results")

    # Mean over all datasets
    mean_df = df.groupby(['mde', 'solver'])[metric_cols].mean().reset_index()
    calib_rows, uncal_rows = collect_rows(mean_df)
    print_table(calib_rows, "MEAN - Best Calib Results")
    print_table(uncal_rows, "MEAN - Best Uncal Results")


if __name__ == '__main__':
    results_df = pd.read_csv('csv_results/d2p_pose_results.csv')
    print_best_only_table(results_df, sort_rows=True)