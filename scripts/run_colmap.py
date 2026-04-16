import subprocess
import os
from pathlib import Path



def run_colmap_extractor(database_path, image_path):
    subprocess.call([
        "colmap",
        "feature_extractor",
        "--database_path",
        database_path,
        "--image_path",
        image_path
    ])

def run_colmap_exhaustive_matcher(database_path, image_path):
    subprocess.call([
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        database_path,
    ])

def run_colmap_mapper(database_path, image_path, model_path):
    subprocess.call([
        "colmap",
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        image_path,
        "--output_path",
        model_path
    ])



if __name__ == '__main__':
    root = Path("/mnt/data/gg/mdrpbench_datasets_collected")
    model_name = "colmap"

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]

    for scene in scenes:
        subsets = sorted(list((root / scene).glob("*")))
        subsets = [subset.stem for subset in subsets if subset.is_dir()]

        for subset in subsets:
            image_path = root / scene / subset / "images"
            database_dir = root / scene / subset / "databases"

            model_path = root / scene / subset / model_name
            model_path.mkdir(parents=True, exist_ok=True)

            database_path = database_dir / "sift_nn_colmap.db"

            if database_path.exists():
                os.remove(database_path)
            database_dir.mkdir(parents=True, exist_ok=True)

            run_colmap_extractor(database_path, image_path)
            run_colmap_exhaustive_matcher(database_path, image_path)
            run_colmap_mapper(database_path, image_path, model_path)
        #     break
        # break
