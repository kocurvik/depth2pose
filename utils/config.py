import copy
import json
import os


def config_iterator(config_path, return_dataset_type=False):
    """Iterate over (name, config[, dataset_type]) of all subsets (scenes) in a dataset config.

    config_path is either a path to a JSON config (see datasets/*.json) or an already loaded config dict.
    """
    if isinstance(config_path, dict):
        dataset_config = copy.deepcopy(config_path)
    else:
        with open(config_path) as f:
            dataset_config = json.load(f)

    for dataset_type, super_config in dataset_config.items():
        for name, config in super_config["subsets"].items():
            config['work_path'] = os.path.join(super_config["work_path"], name)
            config['path'] = os.path.join(super_config["path"], name)

            if "single_scene_subsets" in super_config:
                config["single_scene_subsets"] = super_config["single_scene_subsets"]

            if return_dataset_type:
                yield name, config, dataset_type
            else:
                yield name, config