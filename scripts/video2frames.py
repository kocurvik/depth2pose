import numpy as np
import subprocess
from pathlib import Path


if __name__ == '__main__':
    root = Path("/mnt/data/gg/mdrpbench_datasets_collected/")

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]

    for scene in scenes:
        scene_path = root / scene
        image_path = scene_path / "images"
        if image_path.exists():
            continue

        video_file = scene_path /
