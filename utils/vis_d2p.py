import os
import sys
import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.colors import LinearSegmentedColormap
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise results for the d2p dataset. Results are already computed and sotred in a csv file."
    )
    parser.add_argument("--input-file", type=Path, nargs="+") # path to input file. Can be more than one, if lineplots, it has to be two, one for d2p and one for standard
    parser.add_argument("--output-path", type=Path) # folder to save stuff in 
    parser.add_argument("--iters", type=int, default=1000) # number of ransac iterations to take results from
    parser.add_argument("--new-scene-groups", type=Path, default=None) # if scenes have been regrouped a csv with new scene mappings can be sent in
    parser.add_argument("--coloring-base", type=str, default='rank') # can be rank, maa or inliers. Decides which of these values colors the heatmaps
    parser.add_argument("--solver-type", type=str, default='') # if only results from one solver should be taken into accout. Baseline is always included. Might produce empty plots and warnings
    parser.add_argument("--baseline-mde", type=str, default='none') # which mde to use as baseline in the coloring of the heatmap
    parser.add_argument("--split-ro", action="store_true") # if results from ro should be considered separate. Otherwise, only the mest over all versions are taken
    parser.add_argument("--save-heatmap", action="store_true") # if the heatmap(s) should be saved
    parser.add_argument("--plot-separate-scenes", action="store_true") # if heatmaps for separate scenes should be plotted all in one plot. cannot be used together with lineplot option
    parser.add_argument("--save-lineplot", action="store_true") # if lineplots should be saved. Requires two inlut files
    parser.add_argument("--use-green-ranking", action="store_true") # for the heatmap, i fonly one color (green) should be used. Otherwise green is good, red bad, yellow in the middle
    return parser.parse_args()


# stuff for lineplots




def get_mde_marker(mde, basename=None, backbone=None):

    calib = 'Calib' in mde

    style = {}
    if calib:
        style['markeredgecolor'] = 'black'

    if backbone == 'ViT-S':
        style['marker'] = 'o'
    elif backbone == 'ViT-B':
        style['marker'] = 'v'
    elif backbone == 'ViT-G':
        style['marker'] = 'd'
    elif backbone == 'ConvNext':
        style['marker'] = 'h'
    else:
        style['marker'] = 'p'

    if 'MoGe' in basename:
        style['markerfacecolor'] = '#2ca02c'   # tab green
        if '1' in basename:
            style['markerfacecolor'] = '#98df8a'  # light green

    if 'UniDepth' in basename or 'UniK3D' in basename:
        style['markerfacecolor'] = '#1f77b4'   # tab blue
        if '1' in basename:
            style['markerfacecolor'] = '#aec7e8'  # light blue
        if '3' in basename:
            style['markerfacecolor'] = '#6baed6'  # medium blue (UniK3D)

    if 'Metric3D' in basename:
        style['markerfacecolor'] = '#17becf'   # tab teal

    if 'DepthAnything' in basename:
        style['markerfacecolor'] = '#ff9896'  # light red

    if 'DAv3' in basename:
        style['markerfacecolor'] = '#ff7f7f'   # tab red

        if 'Metric' in mde:
            style['markerfacecolor'] = '#ff4d4d'  # tab orange

    if 'VGGT' == basename:
        style['markerfacecolor'] = '#9467bd'   # tab purple

    if 'Pi3' == basename:
        style['markerfacecolor'] = '#ff7f0e'   # tab orange

    if 'MapAnything' in basename:
        style['markerfacecolor'] = '#e377c2'   # tab pink

    if 'DepthPro' == basename:
        style['markerfacecolor'] = '#8c564b'   # tab brown

    # Unused
    if 'InfiniDepth' == basename:
        style['markerfacecolor'] = '#7f7f7f'   # tab gray

    if 'markerfacecoloralt' in style:
        style['fillstyle'] = 'left'
    else:
        style['fillstyle'] = 'full'

    style['color'] = style['markerfacecolor']
    style['markersize'] = 8

    return style





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
    # Normalize each dataset
    processed_dicts = {
        name: normalize_ranks(d, remove_keys)
        for name, d in dicts.items()
    }

    return processed_dicts

def plot_maa(maa_dicts, output_file, title_ending='',nicknames=''):
    #dataset_names = list(maa_dicts.keys())
    ordered_first = ['sintel', 'scannetpp', 'eth3d', 'lamar']
    remaining = [d for d in maa_dicts if d not in ordered_first]
    # Sort remaining by descending average mAA
    if maa_dicts is not None:
        remaining.sort(
            #key=lambda d: np.mean(list(maa_dicts[d].values())),
            key=lambda d: np.max(list(maa_dicts[d].values())),
            reverse=True
        )
    dataset_names = ordered_first + remaining
    dataset_nicknames = [nicknames[dataset_name] for dataset_name in dataset_names]

    # Find common models AFTER filtering
    models = set.intersection(*(set(d.keys()) for d in maa_dicts.values()))
    models = sorted(models)
    models_nicknames = {
        'DepthAnythingV2': 'DAv3',
        'MapAnything': 'Map\nAnything',
        'MoGeV2': 'MoGeV2',
        'Pi3': 'Pi3',
        'UniK3D': 'UniK3D'
    }

    x = np.arange(len(dataset_names))
 
    plt.figure(figsize=(20, 5))

    style_cycle = get_style_cycle()

    for model in models:
        color, linestyle = next(style_cycle)
        y = [maa_dicts[d][model] for d in dataset_names]

        #plt.plot(x, y, marker='o', linestyle=linestyle, color=color, label=model)
        style = get_mde_marker(model,model)
        plt.plot(x, y, linestyle=linestyle, **style, label=models_nicknames[model])


    split_idx = 4
    plt.axvspan(
        -1.5,
        split_idx - 0.5,
        color='lightgray',
        alpha=0.2,
        zorder=0
    ) 
    plt.axvspan(
        split_idx - 0.5,
        12,
        color='lightblue',
        alpha=0.2,
        zorder=0
    ) 
    plt.axvline(
        x=split_idx - 0.5,
        color='gray',
        linestyle='--',
        linewidth=1
    )
    
    #plt.xticks(x, dataset_names, rotation=90)
    plt.xticks(x, dataset_nicknames)
    plt.yticks(np.asarray([70,80,90]))
    #plt.gca().invert_yaxis()
    #plt.xlabel("Dataset")
    plt.ylim(63,94)
    plt.xlim(-0.3,10.3)
    plt.ylabel("mAA@10")
    #plt.title("Model Rankings " + title_ending)
    
    plt.rcParams.update({        # Use LaTeX to render text
        "font.family": "serif",   
        'font.size': 16,          # base size for everything
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    }) 

    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()

def filter_models(dicts, remove_mdes):
    return {
        category: {
            scene: {
                model: value
                for model, value in models.items()
                if model not in remove_mdes
            }
            for scene, models in scenes.items()
        }
        for category, scenes in dicts.items()
    }
def filter_scenes(dicts, remove_scenes):
    return {
        scene: {
            model: value
            for model, value in models.items()
            if model not in remove_scenes
        }
        for scene, models in dicts.items()
    }

def plot_maa_lineplot(maa_dicts, output_path, plot_with_avg):
    
    # manually prune data
    remove_mdes = ['gt','none','VGGT','Metric3DV2','DepthAnythingV3','MoGeV1','UniDepth2','UniDepth1','DepthPro']
    
    statues_scenes = ['babies_cerny','giant_chair_vltava','kyoto_nakagawahigashiyama_statue','niemand_statue','r2d2_folimanka','yvr_huge_chair','franz_kafka','hanging_girl','memorial_victims_of_communism','prague_castle_statue','spejbl_hurvinek','zlata_kasna_chrt']
    remove_statues = ['babies_cerny','giant_chair_vltava','niemand_statue','r2d2_folimanka','yvr_huge_chair','hanging_girl','prague_castle_statue','spejbl_hurvinek']
    
    vegetation_scenes = ['dresden_1','kyoto_nakagawahigashiyama_tree','overgrown_statue_couple','stromovka_2','stromovka_3','stromovka_7','klamovka_sign1','nadina_domestica','overgrown_statue_pot','stromovka_2b','stromovka_5','tree_orbit_normal1']
    remove_vegetation = ['dresden_1','stromovka_2','stromovka_3','stromovka_7','klamovka_sign1','nadina_domestica','overgrown_statue_pot','tree_orbit_normal1','overgrown_statue_couple']

    nicknames = {
        'sintel': 'Sintel',
        'scannetpp': 'ScanNet\n++',
        'eth3d': 'ETH3D',
        'lamar': 'LaMaR',
        'franz_kafka': 'Franz\nKafka',
        'zlata_kasna_chrt': 'Zlata\nkasna',
        'stromovka_2b': 'Stromovka2b',
        'memorial_victims_of_communism': 'Memorial',
        'stromovka_5': 'Stromovka5',
        'kyoto_nakagawahigashiyama_statue': 'Kyoto\nstatue',
        'kyoto_nakagawahigashiyama_tree': 'Kyoto\ntree'
    }


    maa_dicts = filter_models(maa_dicts, remove_mdes)

    maa_dicts = filter_scenes(maa_dicts,remove_statues+remove_vegetation)

    for category, maa_dict in maa_dicts.items(): 
        save_path = output_path / f"{category}_maa_subset_lineplot.pdf"
        #save_path = output_path / f"{category}_maa_statues_lineplot.png"

        plot_maa(maa_dict, save_path, title_ending=' '+category, nicknames=nicknames)

        '''
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
            avg_maa = {}
            models = set().union(*filtered_dicts)
            for model in models:
                values = [d[model] for d in rank_dicts.values() if model in d]
                avg_ranks[model] = sum(values) / len(values)

            rank_dicts_with_avg = {
                "mean (standard)": avg_ranks,
                **rank_dicts
            }
            plot_rankings(rank_dicts_with_avg, save_path,title_ending=' '+category)
        '''


def plot_rankings(dicts, output_file, title_ending='', maa_dicts=None):
    dataset_names = list(dicts.keys())

    # Find common models AFTER filtering
    models = set.intersection(*(set(d.keys()) for d in dicts.values()))
    models = sorted(models)

    x = np.arange(len(dataset_names))
 
    figwidth = max(10, 1 * len(dataset_names))
    plt.figure(figsize=(figwidth, 16))

    style_cycle = get_style_cycle()

    for model in models:
        color, linestyle = next(style_cycle)
        y = [dicts[d][model] for d in dataset_names]

        plt.plot(x, y, marker='o', linestyle=linestyle, color=color, label=model)

        if maa_dicts is not None:
            for i, dataset in enumerate(dataset_names):
                if dataset in maa_dicts and model in maa_dicts[dataset]:
                    maa_val = maa_dicts[dataset][model]
                    plt.text(
                        x[i],
                        y[i] - 0.1,  # slightly above point (since inverted y-axis)
                        f"{maa_val:.2f}",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color=color
                    )

    plt.xticks(x, dataset_names, rotation=90, ha='center')
    plt.gca().invert_yaxis()
    plt.xlabel("Dataset")
    plt.ylabel("Rank")
    plt.title("Model Rankings " + title_ending)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()

def plot_rankings_paper(dicts, output_file, title_ending='', maa_dicts=None):
    #dataset_names = list(dicts.keys())
    ordered_first = ['sintel', 'scannetpp', 'eth3d', 'lamar']
    remaining = [d for d in dicts if d not in ordered_first]
    random.shuffle(remaining)
    dataset_names = ordered_first + remaining

    # Find common models AFTER filtering
    models = set.intersection(*(set(d.keys()) for d in dicts.values()))
    models = sorted(models)

    x = np.arange(len(dataset_names))
 
    figwidth = max(10, 1 * len(dataset_names))
    plt.figure(figsize=(figwidth, 8))

    style_cycle = get_style_cycle()

    for model in models:
        color, linestyle = next(style_cycle)
        y = [dicts[d][model] for d in dataset_names]

        plt.plot(x, y, marker='o', linestyle=linestyle, color=color, label=model)

        if maa_dicts is not None:
            for i, dataset in enumerate(dataset_names):
                if dataset in maa_dicts and model in maa_dicts[dataset]:
                    maa_val = maa_dicts[dataset][model]
                    plt.text(
                        x[i],
                        y[i] - 0.1,  # slightly above point (since inverted y-axis)
                        f"{maa_val:.2f}",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color=color
                    )

    plt.xticks(x, dataset_names, rotation=90, ha='center')
    plt.gca().invert_yaxis()
    plt.xlabel("Dataset")
    plt.ylabel("Rank")
    plt.title("Model Rankings " + title_ending)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()



def plot_lineplot_paper(rank_dicts, output_path, plot_with_avg, maa_dicts):

    # manually prune data
    remove_mdes = ['gt','none','VGGT','Metric3DV2','DepthAnythingV3','MoGeV1','UniDepth2','UniDepth1','DepthPro']
    
    statues_scenes = ['babies_cerny','giant_chair_vltava','kyoto_nakagawahigashiyama_statue','niemand_statue','r2d2_folimanka','yvr_huge_chair','franz_kafka','hanging_girl','memorial_victims_of_communism','prague_castle_statue','spejbl_hurvinek','zlata_kasna_chrt']
    remove_statues = ['babies_cerny','giant_chair_vltava','niemand_statue','r2d2_folimanka','yvr_huge_chair','hanging_girl','prague_castle_statue','spejbl_hurvinek']
    
    vegetation_scenes = ['dresden_1','kyoto_nakagawahigashiyama_tree','overgrown_statue_couple','stromovka_2','stromovka_3','stromovka_7','klamovka_sign1','nadina_domestica','overgrown_statue_pot','stromovka_2b','stromovka_5','tree_orbit_normal1']
    remove_vegetation = ['dresden_1','stromovka_2','stromovka_3','stromovka_7','klamovka_sign1','nadina_domestica','overgrown_statue_pot','tree_orbit_normal1','overgrown_statue_couple']

    

    maa_dicts = filter_models(maa_dicts, remove_mdes)
    rank_dicts = filter_models(rank_dicts, remove_mdes)

    #maa_dicts = filter_scenes(maa_dicts,statues_scenes+remove_vegetation)
    #rank_dicts = filter_scenes(rank_dicts,statues_scenes+remove_vegetation)

    #maa_dicts = filter_scenes(maa_dicts,vegetation_scenes+remove_statues)
    #rank_dicts = filter_scenes(rank_dicts,vegetation_scenes+remove_statues)

    maa_dicts = filter_scenes(maa_dicts,remove_statues+remove_vegetation)
    rank_dicts = filter_scenes(rank_dicts,remove_statues+remove_vegetation)

    
    for category, rank_dicts in rank_dicts.items(): 
        #save_path = output_path / f"{category}_rank_vegetation_lineplot.png"
        #save_path = output_path / f"{category}_rank_statues_lineplot.png"
        save_path = output_path / f"{category}_rank_subset_lineplot.png"
        

        plot_rankings_paper(rank_dicts, save_path,title_ending=' '+category, maa_dicts=maa_dicts.get(category))


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
            avg_maa = {}
            models = set().union(*filtered_dicts)
            for model in models:
                values = [d[model] for d in rank_dicts.values() if model in d]
                avg_ranks[model] = sum(values) / len(values)

            rank_dicts_with_avg = {
                "mean (standard)": avg_ranks,
                **rank_dicts
            }
            plot_rankings(rank_dicts_with_avg, save_path,title_ending=' '+category)



def plot_lineplot(rank_dicts, output_path, plot_with_avg, maa_dicts = None):
    
    for category, rank_dicts in rank_dicts.items(): 
        save_path = output_path / f"{category}_lineplot.png"

        plot_rankings(rank_dicts, save_path,title_ending=' '+category, maa_dicts=maa_dicts.get(category))


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
            avg_maa = {}
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
        split_ro = False,
        solver_type = ''
    ):
    """
    Returns:
    data[category][solver_group][scene][model] = best_result

    category = vegetation / statue / unclear
    solver_group = calib / uncal (optionally calib_ro, etc.)
    """
       
    df_orig = pd.read_csv(filepath)

    # manually filter out the ambiguous scene
    df_orig = df_orig[df_orig['group']!='ambiguous']

    # update to new groups for the scenes
    if new_scene_groups is not None:
        df = update_group(df_orig,new_scene_groups)
    else:
        df = df_orig

    # filter by iterations
    df = df[df['iters'] == nbr_iters].copy()
    # if a solver type is submitted, use only this, by filtering #gabbi
    if solver_type: #gabbi
        # get solver group
        calib_uncal = get_solver_group(solver_type, calib_solvers, uncal_solvers, False)
        # keep only the chosen solver type and baseline of correct type
        #df = df[
        #    (df['solver'] == solver_type) |
        #    (df['solver'].str.contains('baseline', na=False))
        #].copy()
        if calib_uncal == 'calib':
            df = df[
                (df['solver'] == solver_type) |
                (df['solver'] == 'baseline_calib')
            ].copy()
        elif calib_uncal == 'uncal':
            df = df[
                (df['solver'] == solver_type) |
                (df['solver'] == 'baseline_sf')
            ].copy()
        else:
            print('Problem!')
        #gabbi this should remove everything that is other solves than e.g. calib_ro, making it possible to run everything else as it is, hopefully
    

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

def add_mAA_inliers_rankings_mean(maa_matrix, inliers_matrix, color_matrix, model_names, coloring_base, use_green_ranking, baseline_mde):
    """
    Adds a 'mean' row on top of the matrices.
    - mAA: mean over scenes
    - ranking: recomputed from mean mAA (not averaged ranks)
    - coloring: same logic as elsewhere (relative to 'none')
    """

    # --- compute mean mAA ---
    mean_maa = np.nanmean(maa_matrix, axis=0)
    mean_inliers = np.nanmean(inliers_matrix, axis=0)

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

    if baseline_mde in model_names:
        baseline_idx = model_names.index(baseline_mde) 
        baseline_rank = mean_ranks[baseline_idx] 
        baseline_maa = mean_maa[baseline_idx] 
        baseline_inliers = mean_inliers[baseline_idx] 
        max_rank = np.nanmax(mean_ranks) 

        for j in range(len(mean_maa)):
            if np.isnan(mean_ranks[j]):
                continue

            if coloring_base == 'rank':
                rank = mean_ranks[j]
                if use_green_ranking:
                    mean_color[j] = 2 * (max_rank - rank) / (max_rank - 1 + 1e-6) -1
                else:
                    if rank < baseline_rank:
                        mean_color[j] = (baseline_rank - rank) / (baseline_rank - 1 + 1e-6)
                    elif rank > baseline_rank:
                        mean_color[j] = -(rank - baseline_rank) / (max_rank - baseline_rank + 1e-6)
                    else:
                        mean_color[j] = 0

            elif coloring_base == 'maa':
                if use_green_ranking:
                    mean_color[j] = mean_maa[j] - baseline_maa
                else:
                    mean_color[j] = mean_maa[j] - baseline_maa

            elif coloring_base == 'inliers':
                if use_green_ranking:
                    mean_color[j] = mean_inliers[j]
                else:
                    mean_color[j] = mean_inliers[j] - baseline_inliers



    # --- prepend row ---
    new_maa = np.vstack([mean_maa, maa_matrix])
    new_color = np.vstack([mean_color, color_matrix])
    new_inliers = np.vstack([mean_inliers, inliers_matrix])

    # obs. This assumes results for all models. Will fail otherwise
    mean_ranks_dict = {model_name: rank for model_name, rank in zip(model_names, mean_ranks)}
    mean_maa_dict = {model_name: maa for model_name, maa in zip(model_names, mean_maa)}

    return new_maa, new_inliers, new_color, mean_ranks_dict, mean_maa_dict

def plot_heatmaps(data, rankings, output_path, mode, coloring_base, use_green_ranking=False, save_heatmap=True, baseline_mde='none', plot_separate_scenes = False):

    # as input, add input 1) plot_heatmap and 2) plot_lineplot
    # or just make copy for lineplot? and there you then have to submit more data. More clean though?
    # or first call this, return scene_ranks and then call plot linemap after
    # we still need option to not plot/save heatmaps then, but it would be called separately for lineplot which is more clean

    unique_solver_groups = {sg for solver_groups in data.values() for sg in solver_groups.keys()}
    rank_dicts = {sg: {} for sg in unique_solver_groups}
    maa_dicts = {sg: {} for sg in unique_solver_groups}

    for category, solver_groups in data.items():

        for solver_group, scenes in solver_groups.items():

            scene_names = sorted(scenes.keys())
            all_models = {m for s in scenes.values() for m in s.keys()}
            normal_models = sorted(m for m in all_models if m not in ["gt", "none"])
            model_names = normal_models + ["none", "gt"]

            color_matrix = np.full((len(scene_names), len(model_names)), np.nan)
            maa_matrix = np.full((len(scene_names), len(model_names)), np.nan)
            inliers_matrix = np.full((len(scene_names), len(model_names)), np.nan)

            for i, scene in enumerate(scene_names):

                scene_ranks = rankings[category][solver_group][scene]

                if "none" not in scene_ranks:
                    print('oops none')
                    #breakpoint()
                    continue

                baseline_rank = scene_ranks[baseline_mde] 
                max_rank = max(scene_ranks.values())

                baseline_maa = data[category][solver_group][scene][baseline_mde]['mAA']
                baseline_inliers = data[category][solver_group][scene][baseline_mde]['inliers']

                for j, model in enumerate(model_names):
                    if model in scene_ranks:

                        rank = scene_ranks[model]
                        mAA = data[category][solver_group][scene][model]["mAA"]
                        inliers = data[category][solver_group][scene][model]["inliers"]
                        
                        maa_matrix[i, j] = mAA
                        inliers_matrix[i,j] = inliers

                        if coloring_base == 'rank':
                            if use_green_ranking:
                                color_matrix[i, j] = 2 * (max_rank - rank) / (max_rank - 1 + 1e-6) - 1

                            else:
                                if rank < baseline_rank:
                                    color_matrix[i, j] = (baseline_rank - rank) / (baseline_rank - 1 + 1e-6)
                                elif rank > baseline_rank:
                                    color_matrix[i, j] = -(rank - baseline_rank) / (max_rank - baseline_rank + 1e-6)
                                else:
                                    color_matrix[i, j] = 0
                            vmin = -1
                            vmax = 1
                        elif coloring_base == 'maa':
                            if use_green_ranking:
                                color_matrix[i, j] = mAA - baseline_maa
                            else:
                                color_matrix[i, j] = mAA - baseline_maa
                            vmin = -40
                            vmax = 5
                        elif coloring_base == 'inliers':
                            if use_green_ranking:
                                color_matrix[i, j] = inliers
                                vmin = inliers_matrix.min()
                                vmax = inliers_matrix.max()
                            else:
                                color_matrix[i, j] = inliers - baseline_inliers
                                vmin = -30
                                vmax = 10
                
                if mode == 'standard':
                    rank_dicts[solver_group][scene] = normalize_ranks(scene_ranks, remove_keys=["gt", "none"])
                    maa_dicts[solver_group][scene] = {key: data[category][solver_group][scene][key]['mAA'] for key in data[category][solver_group][scene].keys()}
                if mode == 'd2p' and plot_separate_scenes:
                    rank_dicts[solver_group][scene] = normalize_ranks(scene_ranks, remove_keys=["gt", "none"])
                    maa_dicts[solver_group][scene] = {key: data[category][solver_group][scene][key]['mAA'] for key in data[category][solver_group][scene].keys()}

    
            # this is where we have "solver_group" (calib/uncal/calib_ro/uncal_ro)
            # this is also where we have cetegory (vegetation/ambiguous/statues)
            # heatmap plotting is also done here, so we do not necessarily need to fix stuffhere, just in this level of loop
            # we also have "model_names"
            # from add_mAA_mean we can return ranks, and together with model_names create dict
            # the dict can then be added into the calib_dicts of uncal_discts
            # maybe only do this option without ro? or also with? just four if..ask chatgpt
            
            # gabbi fix the addition of mAA mean, add inlier_mean instead
            maa_matrix, inliers_matrix, color_matrix, mean_ranks, mean_maa = add_mAA_inliers_rankings_mean(
                maa_matrix, inliers_matrix, color_matrix, model_names, coloring_base, use_green_ranking, baseline_mde
            )
            scene_names = ["mean"] + scene_names

            if mode == 'd2p' and not plot_separate_scenes:
                # this is actually only needed for ranks, to not get "holes" in ranks from removing gt and none, not needed for other metrics
                rank_dicts[solver_group][category] = normalize_ranks(mean_ranks, remove_keys=["gt", "none"])
                maa_dicts[solver_group][category] = {k: v for k, v in mean_maa.items() if k not in ["gt", "none"]}

            if coloring_base == 'rank' or coloring_base == 'maa':
                annot_matrix = maa_matrix
            elif coloring_base == 'inliers':
                annot_matrix = inliers_matrix

            if save_heatmap:
                plt.figure(figsize=(12, 6))

                mask = np.isnan(color_matrix)

                if use_green_ranking:
                    sns.heatmap(
                        color_matrix,
                        mask=mask,
                        xticklabels=model_names,
                        yticklabels=scene_names,
                        cmap="Greens", #"RdYlGn",# th 
                        vmin=vmin,
                        vmax=vmax,
                        center=(vmin+vmax)/2,
                        annot=annot_matrix,
                        fmt=".1f",
                        cbar=True
                    )
                    # change colorscale ticks from -1 to 1 to rankings
                    cbar = plt.gca().collections[0].colorbar

                    cbar.set_ticks([vmin, vmax])

                    if coloring_base == 'rank': 
                        cbar.set_ticklabels([
                            "Lowest ranking",
                            "Highest ranking"
                        ])
                    elif coloring_base == 'maa' or coloring_base == 'inliers':
                         cbar.set_ticklabels([
                            vmin,
                            vmax
                        ])                       


                else:
                    sns.heatmap(
                        color_matrix,
                        mask=mask,
                        xticklabels=model_names,
                        yticklabels=scene_names,
                        cmap="RdYlGn",
                        vmin=vmin,
                        vmax=vmax,
                        center=0,
                        annot=annot_matrix,
                        fmt=".1f",
                        cbar=True
                    )
                    # change colorscale ticks from -1 to 1 to rankings
                    cbar = plt.gca().collections[0].colorbar

                    cbar.set_ticks([vmin, 0, vmax])
                    if coloring_base == 'rank':
                        cbar.set_ticklabels([
                            "Lowest ranking",
                            "Baseline",
                            "Highest ranking"
                        ])
                    elif coloring_base == 'maa' or coloring_base == 'inliers':
                        cbar.set_ticklabels([
                            vmin,
                            0,
                            vmax
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

    return rank_dicts, maa_dicts



if __name__ == "__main__":
    args = parse_args()
    calib_solvers = ['calib', 'calib_shift', 'baseline_calib']
    uncal_solvers = ['mdecalib', 'mdecalib_shift', 'sf', 'sf_shift', 'vf', 'vf_shift', 'baseline_sf']

 #   if args.plot_separate_scenes and args.save_lineplot:
 #       raise ValueError("You have to choose between heatmaps for separate scenes and lineplots")
    if args.save_lineplot and len(args.input_file) < 2:
        raise ValueError("For the lineplots you need to input a csv to both d2p and to standard results")
    if args.coloring_base not in ['rank', 'maa', 'inliers']:
        raise ValueError("Coloring base has to be rank, maa or inliers")
    if args.solver_type.endswith('ro') and not args.split_ro:
        raise ValueError("To only use an ro solver, you need to use --split-ro")
    
    os.makedirs(args.output_path, exist_ok=True)
    rank_dicts = {}
    maa_dicts = {}

    for input_file in args.input_file:
        mode = 'd2p' if 'd2p' in str(input_file) else 'standard' if 'standard' in str(input_file) else None
        data = parse_csv_file(
            input_file,
            calib_solvers,
            uncal_solvers,
            new_scene_groups = args.new_scene_groups,
            nbr_iters=args.iters,
            split_ro = args.split_ro,
            solver_type = args.solver_type
        ) #default is using 1000 iterations, can be inputted

        rankings = compute_rankings(data)

        if not args.plot_separate_scenes:
            mode_rank_dicts, mode_maa_dicts = plot_heatmaps(data, rankings, args.output_path, mode, args.coloring_base, args.use_green_ranking, args.save_heatmap, args.baseline_mde, plot_separate_scenes = args.plot_separate_scenes)
        
        else:

            data_merged = {'d2p': {}}

            for old_group, inner_dict in data.items():
                for solver_group, scenes in inner_dict.items():
                    if solver_group not in data_merged['d2p']:
                        data_merged['d2p'][solver_group] = {}
                    data_merged['d2p'][solver_group].update(scenes)

            rankings_merged = compute_rankings(data_merged)

            mode_rank_dicts, mode_maa_dicts = plot_heatmaps(data_merged, rankings_merged, args.output_path, mode, args.coloring_base, args.use_green_ranking, args.save_heatmap, args.baseline_mde, plot_separate_scenes = args.plot_separate_scenes)

        if args.save_lineplot:
            for solver_group, inner_dict in mode_rank_dicts.items():
                if solver_group not in rank_dicts:
                    rank_dicts[solver_group] = {}
                rank_dicts[solver_group].update(inner_dict)
            for solver_group, inner_dict in mode_maa_dicts.items():
                if solver_group not in maa_dicts:
                    maa_dicts[solver_group] = {}
                maa_dicts[solver_group].update(inner_dict)

    if args.save_lineplot:   
        plot_with_avg = True
        #plot_lineplot(rank_dicts, args.output_path, plot_with_avg, maa_dicts)
        plot_lineplot_paper(rank_dicts, args.output_path, plot_with_avg, maa_dicts)
        plot_maa_lineplot(maa_dicts, args.output_path, plot_with_avg)
    