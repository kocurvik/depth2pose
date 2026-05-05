import pandas as pd
import argparse
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise results for the d2p dataset. Results are already computed and sotred in a csv file."
    )
    parser.add_argument("--input-file", type=Path, nargs="+")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--new-scene-groups", type=Path, default=None)
    parser.add_argument("--split-ro", action="store_true")
    parser.add_argument("--save-heatmap", action="store_true")
    parser.add_argument("--plot-separate-scenes", action="store_true")
    parser.add_argument("--save-lineplot", action="store_true")
    parser.add_argument("--use-green-ranking", action="store_true") # for the heatmap
    return parser.parse_args()


# stuff for lineplots


def get_style_cycle():
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['-', '--', '-.', ':']

    # Create all combinations
    return itertools.chain.from_iterable(
        [(c, ls) for c in colors] for ls in linestyles
    )

def normalize_ranks(d, remove_keys=None):
    if remove_keys is None:
        remove_keys = []
    # 1. filter
    filtered = {k: v for k, v in d.items() if k not in remove_keys}

    # 2. sort by original rank
    sorted_items = sorted(filtered.items(), key=lambda x: x[1])

    # 3. reassign ranks (1..N)
    return {k: i + 1 for i, (k, _) in enumerate(sorted_items)}

def get_common_models(dicts):
    common = set.intersection(*(set(d.keys()) for d in dicts.values()))
    return sorted(common)

def exclude_special(dicts, remove_keys=["gt", "none"]):
    breakpoint()
    # Normalize each dataset
    processed_dicts = {
        name: normalize_ranks(d, remove_keys)
        for name, d in dicts.items()
    }

    return processed_dicts

def plot_rankings(dicts, output_file, title_ending=''):
    dataset_names = list(dicts.keys())

    # Find common models AFTER filtering
    models = set.intersection(*(set(d.keys()) for d in dicts.values()))
    models = sorted(models)

    x = np.arange(len(dataset_names))

    plt.figure(figsize=(10, 6))

    style_cycle = get_style_cycle()

    for model in models:
        color, linestyle = next(style_cycle)
        y = [dicts[d][model] for d in dataset_names]
        
        plt.plot(x, y, marker='o', linestyle=linestyle, color=color, label=model)

    plt.xticks(x, dataset_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Dataset")
    plt.ylabel("Rank")
    plt.title("Model Rankings " + title_ending)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def plot_lineplot(rank_dicts, output_path, plot_with_avg):
    
    for category, rank_dicts in rank_dicts.items(): 
        save_path = output_path / f"{category}_lineplot.png"

        plot_rankings(rank_dicts, save_path,title_ending=' '+category)


    if plot_with_avg:
        save_path = output_path / f"{category}_lineplot_avg.png"
        # compute average ranks
        selected_datasets = {'eth3d', 'lamar', 'scannetpp', 'sintel'}

        filtered_dicts =  [
            rank_dicts[k]
            for k in selected_datasets
            if k in rank_dicts
        ]
        avg_ranks = {}
        models = set().union(*filtered_dicts)
        for model in models:
            values = [d[model] for d in rank_dicts.values() if model in d]
            avg_ranks[model] = sum(values) / len(values)

        rank_dicts_with_avg = {
            "mean (standard)": avg_ranks,
            **rank_dicts
        }
        plot_rankings(rank_dicts_with_avg, save_path,title_ending=' '+category)


# below_: stuff for heatmaps
def get_solver_group(s, calib_solvers, uncal_solvers, split_ro):
    if any(s.startswith(base) for base in calib_solvers):
        base = "calib"
    elif any(s.startswith(base) for base in uncal_solvers):
        base = "uncal"
    else:
        return None
    if split_ro:
        is_baseline = s.startswith("baseline_")
        if is_baseline: 
            return f"{base}_both"
        if "_ro" in s:
            return f"{base}_ro" 
        else:
            #return f"{base}_not_ro"
            return f"{base}"
    
    return base

# from utils/tables.py
def build_variant_dict(sub_df):
    variant_dict = {}
    for mde_name in sub_df['mde'].unique():
        base_name = mde_name.split('-')[0].split('Calib')[0]
        #variant_dict.setdefault(base_name, []).append(mde_name)
        if base_name not in variant_dict:
            variant_dict[base_name] = []
        variant_dict[base_name].append(mde_name)
    return variant_dict

# from utils/tables.py
def find_best(sub_df, variants, solver_group, metric_cols):
    #if 'gt' in variants:
    sub = sub_df[sub_df['mde'].isin(variants)]
    sub = sub[sub['solver_group'] == solver_group]
    if sub.empty:
        print("find_best empty!", variants, solver_group)
        return None
    best_row = sub.loc[sub['pose_mAA_10'].idxmax()]
    return best_row['mde'], best_row['solver'], best_row[metric_cols]

def build_data_from_df(df, metric_cols):

    data = {}
    #breakpoint()
    for category in df['group'].unique():
        # this for loop iterates over different groups/categories, likely 2 or 3 different
        # group_df is a df that contains only the parts of df with this specific group
        category_df = df[df['group'] == category]
        data[category] = {}

        for solver_group in category_df['solver_group'].unique():
            # loop over solver groups, either calib/uncal or calib_ro/calib_not_ro/uncal_ro/uncal_not_ro
            # get only solvers belonging to this group
            solver_group_df = category_df[category_df['solver_group']==solver_group]

            for dataset in solver_group_df['dataset'].unique():
                # this for loop iterates over different datasets/scenes. All these will be saved together in data
                # sub_df is a df that contains only parts with the specific group and specific dataset
                sub_df = solver_group_df[solver_group_df['dataset'] == dataset]

                # variant_dict contains the mde as key and different variants (e.g. using different backbones) and values
                # also contains 'gt' and 'none'
                variant_dict = build_variant_dict(sub_df)

                # dataset_df is a version of sub_df that removes the group, dataset and iters columns, since these are all the same
                dataset_df = (
                    sub_df.groupby(['mde', 'solver', 'solver_group'])[metric_cols]
                    .mean()
                    .reset_index()
                )

                for base_mde, variants in variant_dict.items():
                    # iterate over the different mde:s used
                    
                    """
                    #best = find_best(dataset_df, variants_use, solver_subset, metric_cols)
                    print("----")
                    print("solver_group:", solver_group)
                    print("variants:", variants[:3])
                    print("dataset_df solver_groups:", dataset_df['solver_group'].unique())
                    print("dataset_df mde sample:", dataset_df['mde'].unique()[:5])
                    """
                    # find the mde variant that performs the best 
                    best = find_best(sub_df, variants, solver_group, metric_cols)

                    '''
                    if best:
                        _, _, metrics = best
                        data[category].setdefault(solver_group, {}) \
                                    .setdefault(dataset, {})[base_mde] = {
                            'mAA': float(metrics['pose_mAA_10']),
                            'runtime': float(metrics['mean_mde_runtime']),
                            'inliers': float(metrics['mean_inliers']),
                        }
                    '''
                    if best:
                        _, _, metrics = best
                        # just make sure all levels of dict exist
                        if category not in data:
                            data[category] = {}
                        if solver_group not in data[category]:
                            data[category][solver_group] = {}
                        if dataset not in data[category][solver_group]:
                            data[category][solver_group][dataset] = {}

                        # final assignment
                        data[category][solver_group][dataset][base_mde] = {
                            'mAA': float(metrics['pose_mAA_10']),
                            'runtime': float(metrics['mean_mde_runtime']),
                            'inliers': float(metrics['mean_inliers']),
                        }
            # manually add the no_depth, once for each solver_group
            # breakpoint()
            #dataset_df[dataset_df['solver'] == 'baseline_calib']


    return data                 

def update_group(df, scene_groups_path):

    print("Before:")
    print(df[["dataset", "group"]].drop_duplicates().sort_values("dataset"))

    new_groups = pd.read_csv(scene_groups_path)
    mapping = dict(zip(new_groups['dataset'], new_groups['group']))
    df['group'] = df['dataset'].map(mapping).fillna(df['group'])

    print("\nAfter:")
    print(df[["dataset", "group"]].drop_duplicates().sort_values("dataset"))

    return df

# parse csv
def parse_csv_file(
        filepath, 
        calib_solvers,
        uncal_solvers,
        new_scene_groups,
        nbr_iters = 1000,
        split_ro = False
    ):
    """
    Returns:
    data[category][solver_group][scene][model] = best_result

    category = vegetation / statue / unclear
    solver_group = calib / uncal (optionally calib_ro, etc.)
    """
       
    df_orig = pd.read_csv(filepath)

    # update to new groups for the scenes
    if new_scene_groups is not None:
        df = update_group(df_orig,new_scene_groups)
    else:
        df = df_orig

    # filter by iterations
    df = df[df['iters'] == nbr_iters].copy()
    
    df['solver_group'] = df['solver'].apply(
        lambda s: get_solver_group(s, calib_solvers, uncal_solvers, split_ro)
    )
    if split_ro:
        # duplicate baseline rows into both _ro and _not_ro
        baseline_df = df[df['solver_group'].str.endswith('_both')].copy()

        if not baseline_df.empty:
            df_ro = baseline_df.copy()
            df_ro['solver_group'] = df_ro['solver_group'].str.replace('_both', '_ro')

            #df_not_ro = baseline_df.copy()
            #df_not_ro['solver_group'] = df_not_ro['solver_group'].str.replace('_both', '_not_ro')
            df_not_ro = baseline_df.copy()
            df_not_ro['solver_group'] = df_not_ro['solver_group'].str.replace('_both', '')

            # remove original "_both"
            df = df[~df['solver_group'].str.endswith('_both')]

            # add duplicated rows
            df = pd.concat([df, df_ro, df_not_ro], ignore_index=True)
    # remove unused rows
    df = df[df['solver_group'].notna()]

    metric_cols = ["pose_mAA_10", "mean_mde_runtime", "mean_inliers"]

    data = build_data_from_df(df, metric_cols)

    return data

def compute_rankings(data):
    rankings = {}

    for category, solver_groups in data.items():
        rankings[category] = {}

        for solver_group, scenes in solver_groups.items():
            rankings[category][solver_group] = {}

            for scene, methods in scenes.items():
                rankings[category][solver_group][scene] = {}

                sorted_methods = sorted(
                    methods.items(),
                    key=lambda x: x[1]['mAA'],
                    reverse=True
                )

                for rank, (model, _) in enumerate(sorted_methods, start=1):
                    rankings[category][solver_group][scene][model] = rank

    return rankings

def add_mAA_mean(maa_matrix, color_matrix, model_names,use_green_ranking):
    """
    Adds a 'mean' row on top of the matrices.
    - mAA: mean over scenes
    - ranking: recomputed from mean mAA (not averaged ranks)
    - coloring: same logic as elsewhere (relative to 'none')
    """

    # --- compute mean mAA ---
    mean_maa = np.nanmean(maa_matrix, axis=0)

    # --- compute rankings (higher mAA = better) ---
    valid_mask = ~np.isnan(mean_maa)
    valid_values = mean_maa[valid_mask]

    # argsort descending
    order = np.argsort(-valid_values)
    ranks = np.empty_like(order)
    # first value is column of highest ranked method, etc
    ranks[order] = np.arange(1, len(valid_values) + 1)

    mean_ranks = np.full_like(mean_maa, np.nan)
    mean_ranks[valid_mask] = ranks # here mean ranks will just be ranks? But not if not all are valie

    # --- compute color row (relative to 'none') ---
    mean_color = np.full_like(mean_maa, np.nan)

    if "none" in model_names:
        none_idx = model_names.index("none")
        none_rank = mean_ranks[none_idx]
        max_rank = np.nanmax(mean_ranks)

        for j in range(len(mean_maa)):
            if np.isnan(mean_ranks[j]):
                continue

            rank = mean_ranks[j]
            if use_green_ranking:
                mean_color[j] = 2 * (max_rank - rank) / (max_rank - 1 + 1e-6) -1
            else:
                if rank < none_rank:
                    mean_color[j] = (none_rank - rank) / (none_rank - 1 + 1e-6)
                elif rank > none_rank:
                    mean_color[j] = -(rank - none_rank) / (max_rank - none_rank + 1e-6)
                else:
                    mean_color[j] = 0

    # --- prepend row ---
    new_maa = np.vstack([mean_maa, maa_matrix])
    new_color = np.vstack([mean_color, color_matrix])

    # obs. This assumes results for all models. Will fail otherwise
    mean_ranks_dict = {model_name: rank for model_name, rank in zip(model_names, mean_ranks)}

    return new_maa, new_color, mean_ranks_dict

def plot_heatmaps(data, rankings, output_path, mode, use_green_ranking=False, save_heatmaps=True):

    # as input, add input 1) plot_heatmap and 2) plot_lineplot
    # or just make copy for lineplot? and there you then have to submit more data. More clean though?
    # or first call this, return scene_ranks and then call plot linemap after
    # we still need option to not plot/save heatmaps then, but it would be called separately for lineplot which is more clean

    unique_solver_groups = {sg for solver_groups in data.values() for sg in solver_groups.keys()}
    rank_dicts = {sg: {} for sg in unique_solver_groups}
    for category, solver_groups in data.items():

        for solver_group, scenes in solver_groups.items():

            scene_names = sorted(scenes.keys())
            all_models = {m for s in scenes.values() for m in s.keys()}
            normal_models = sorted(m for m in all_models if m not in ["gt", "none"])
            model_names = normal_models + ["none", "gt"]

            color_matrix = np.full((len(scene_names), len(model_names)), np.nan)
            maa_matrix = np.full((len(scene_names), len(model_names)), np.nan)

            for i, scene in enumerate(scene_names):

                scene_ranks = rankings[category][solver_group][scene]

                if "none" not in scene_ranks:
                    print('oops none')
                    #breakpoint()
                    continue

                none_rank = scene_ranks["none"]
                max_rank = max(scene_ranks.values())

                for j, model in enumerate(model_names):
                    if model in scene_ranks:

                        rank = scene_ranks[model]
                        mAA = data[category][solver_group][scene][model]["mAA"]

                        maa_matrix[i, j] = mAA
                        if use_green_ranking:
                            color_matrix[i, j] = 2 * (max_rank - rank) / (max_rank - 1 + 1e-6) - 1

                        else:
                            if rank < none_rank:
                                color_matrix[i, j] = (none_rank - rank) / (none_rank - 1 + 1e-6)
                            elif rank > none_rank:
                                color_matrix[i, j] = -(rank - none_rank) / (max_rank - none_rank + 1e-6)
                            else:
                                color_matrix[i, j] = 0
                
                if mode == 'standard':
                    rank_dicts[solver_group][scene] = normalize_ranks(scene_ranks, remove_keys=["gt", "none"])
    
            # this is where we have "solver_group" (calib/uncal/calib_ro/uncal_ro)
            # this is also where we have cetegory (vegetation/ambiguous/statues)
            # heatmap plotting is also done here, so we do not necessarily need to fix stuffhere, just in this level of loop
            # we also have "model_names"
            # from add_mAA_mean we can return ranks, and together with model_names create dict
            # the dict can then be added into the calib_dicts of uncal_discts
            # maybe only do this option without ro? or also with? just four if..ask chatgpt
            
            maa_matrix, color_matrix, mean_ranks = add_mAA_mean(
                maa_matrix, color_matrix, model_names, use_green_ranking
            )
            scene_names = ["mean"] + scene_names

            if mode == 'd2p':
                rank_dicts[solver_group][category] = normalize_ranks(mean_ranks, remove_keys=["gt", "none"])

            if save_heatmaps:
                plt.figure(figsize=(12, 6))

                mask = np.isnan(color_matrix)


                if use_green_ranking:
                    sns.heatmap(
                        color_matrix,
                        mask=mask,
                        xticklabels=model_names,
                        yticklabels=scene_names,
                        cmap="Greens",
                        vmin=-1,
                        vmax=1,
                        center=0,
                        annot=maa_matrix,
                        fmt=".1f",
                        cbar=True
                    )
                    # change colorscale ticks from -1 to 1 to rankings
                    cbar = plt.gca().collections[0].colorbar

                    cbar.set_ticks([-1, 1])
                    cbar.set_ticklabels([
                        "Lowest ranking",
                        "Highest ranking"
                    ])

                else:
                    sns.heatmap(
                        color_matrix,
                        mask=mask,
                        xticklabels=model_names,
                        yticklabels=scene_names,
                        cmap="RdYlGn",
                        vmin=-1,
                        vmax=1,
                        center=0,
                        annot=maa_matrix,
                        fmt=".1f",
                        cbar=True
                    )
                    # change colorscale ticks from -1 to 1 to rankings
                    cbar = plt.gca().collections[0].colorbar

                    cbar.set_ticks([-1, 0, 1])
                    cbar.set_ticklabels([
                        "Lowest ranking",
                        "Baseline",
                        "Highest ranking"
                    ])


                # vertical line before 'none'
                if "none" in model_names:
                    idx = model_names.index("none")
                    plt.axvline(idx, color="white", linewidth=2)

                # horizontal line below 'mean'
                plt.axhline(1, color="white", linewidth=2)

                plt.title(f"{category} - {solver_group}")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                save_path = output_path / f"{category}_{solver_group}_heatmap.png"
                plt.savefig(save_path)
                plt.close()

    return rank_dicts



if __name__ == "__main__":
    args = parse_args()
    calib_solvers = ['calib', 'calib_shift', 'baseline_calib']
    uncal_solvers = ['mdecalib', 'mdecalib_shift', 'sf', 'sf_shift', 'vf', 'vf_shift', 'baseline_sf']

    if args.plot_separate_scenes and args.save_lineplot:
        raise ValueError("You have to choose between heatmaps for separate scenes and lineplots")
    if args.save_lineplot and len(args.input_file) < 2:
        raise ValueError("For the lineplots you need to input a csv to both d2p and to standard results")

    os.makedirs(args.output_path, exist_ok=True)
    rank_dicts = {}

    for input_file in args.input_file:
        mode = 'd2p' if 'd2p' in str(input_file) else 'standard' if 'standard' in str(input_file) else None
        data = parse_csv_file(
            input_file,
            calib_solvers,
            uncal_solvers,
            new_scene_groups = args.new_scene_groups,
            nbr_iters=args.iters,
            split_ro = args.split_ro
        ) #default is using 1000 iterations, can be inputted
        
        rankings = compute_rankings(data)

        if not args.plot_separate_scenes:
            mode_dicts = plot_heatmaps(data, rankings, args.output_path, mode, args.use_green_ranking, args.save_heatmap)
        
        elif args.plot_separate_scenes:

            data_merged = {'d2p': {}}

            for old_group, inner_dict in data.items():
                for solver_group, scenes in inner_dict.items():
                    if solver_group not in data_merged['d2p']:
                        data_merged['d2p'][solver_group] = {}
                    data_merged['d2p'][solver_group].update(scenes)

            rankings_merged = compute_rankings(data_merged)

            _ = plot_heatmaps(data_merged, rankings_merged, args.output_path, mode, args.use_green_ranking, args.save_heatmap)

        if args.save_lineplot:
            for solver_group, inner_dict in mode_dicts.items():
                if solver_group not in rank_dicts:
                    rank_dicts[solver_group] = {}
                rank_dicts[solver_group].update(inner_dict)

    if args.save_lineplot:   
        plot_with_avg = True
        plot_lineplot(rank_dicts, args.output_path, plot_with_avg)
    