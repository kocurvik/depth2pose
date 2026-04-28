import json
import os


def config_iterator(config_path):
    with open(config_path) as f:
        dataset_config = json.load(f)

    for _, super_config in dataset_config.items():
        for name, config in super_config["subsets"].items():
            config['work_path'] = os.path.join(super_config["work_path"], name)
            config['path'] = os.path.join(super_config["path"], name)

            if "single_scene_subsets" in super_config:
                config["single_scene_subsets"] = super_config["single_scene_subsets"]

            yield name, config