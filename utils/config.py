import json
import os


def config_iterator(config_path, return_dataset_type=False):
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