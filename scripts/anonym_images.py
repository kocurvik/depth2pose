import subprocess
from pathlib import Path


"""
Installation instructions to use easyanon

pip install hatchling
pip install git+https://github.com/spatial-intelligence-group/easy_anon.git --no-build-isolation
"""

img_extensions = ['.png', '.jpg', '.jpeg', 'heif', 'heic']

def mask_out_image(image_path, mask_save_dir, image_save_dir):
    subprocess.call([
        "easy-mask",
        image_path,
        mask_save_dir,
        "--labels",
        "person",

    ])
    subprocess.call([
        "easy-anon",
        image_path,
        mask_save_dir,
        image_save_dir,
        "--infill_mode",
        "single_color",

    ])


def run_on_scenes(root: Path, scenes: list[str], selected_scene: str | None = None):
    for scene in scenes:
        if selected_scene is not None and scene != selected_scene:
            continue
        print(f"Running on {scene}")
        scene_dir = root / scene
        image_dir = scene_dir / "images"
        mask_save_dir = scene_dir / "masks"
        image_save_dir = scene_dir / "images_masked"
        mask_save_dir.mkdir(parents=True, exist_ok=True)
        image_save_dir.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            continue

        image_paths = sorted([x for x in image_dir.iterdir() if x.suffix.lower() in img_extensions])

        for image_path in image_paths:
            mask_out_image(image_path, mask_save_dir, image_save_dir)

        if selected_scene is not None:
            break


if __name__ == '__main__':
    root = Path("/mnt/data/gg/mdrpbench_datasets_collected/")
    scene_name = "biden_haus_nowhere"

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]
    run_on_scenes(root, scenes, scene_name)
