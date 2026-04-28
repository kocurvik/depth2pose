import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.colmap_utils import Camera, Image, read_model, write_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RealityScan undistorted reconstructions to a more COLMAP-style layout."
    )
    parser.add_argument(
        "scene_names",
        nargs="+",
        help="One or more scene folder names to process.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory that contains the scene folders.",
    )
    parser.add_argument(
        "--overwrite-old-data",
        action="store_true",
        default=False,
        help="Overwrite realityscan_gt/undistorted instead of writing undistorted_processed. Default: False.",
    )
    return parser.parse_args()


def get_undistorted_paths(scene_dir: Path, overwrite_old_data: bool):
    realityscan_dir = scene_dir / "realityscan_gt"
    undistorted_dir = realityscan_dir / "undistorted"
    input_dir = undistorted_dir if overwrite_old_data else realityscan_dir / "undistorted_orig"
    output_dir = undistorted_dir

    input_images_dir = input_dir / "images"
    input_sparse_dir = input_dir / "sparse"
    output_images_dir = output_dir / "images"
    output_sparse_dir = output_dir / "sparse"

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "undistorted_dir": undistorted_dir,
        "input_images_dir": input_images_dir,
        "input_sparse_dir": input_sparse_dir,
        "output_images_dir": output_images_dir,
        "output_sparse_dir": output_sparse_dir,
    }


def extract_image_name(path_string: str):
    return path_string.strip().replace("\\", "/").split("/")[-1]


def build_undistorted_to_original_name_map(scene_dir: Path, scene_name: str, undistorted_images_dir: Path):
    distorted_images_dir = scene_dir / "realityscan_gt" / "distorted" / "images"

    original_list_path = distorted_images_dir / f"{scene_name}_original.imagelist"
    undistorted_list_path = undistorted_images_dir / f"{scene_name}_undistorted.imagelist"

    original_lines = original_list_path.read_text(encoding="utf-8").splitlines()
    undistorted_lines = undistorted_list_path.read_text(encoding="utf-8").splitlines()

    original_names = [extract_image_name(line) for line in original_lines if line.strip()]
    undistorted_names = [extract_image_name(line) for line in undistorted_lines if line.strip()]

    if len(original_names) != len(undistorted_names):
        raise ValueError(
            f"Image list length mismatch for scene '{scene_name}': "
            f"{len(original_names)} original vs {len(undistorted_names)} undistorted."
        )

    return dict(zip(undistorted_names, original_names))


def prepare_output_images_dir(paths, overwrite_old_data: bool):
    output_images_dir = paths["output_images_dir"]
    if output_images_dir.exists() and not overwrite_old_data:
        shutil.rmtree(output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)


def prepare_input_output_layout(paths, overwrite_old_data: bool):
    if overwrite_old_data:
        return

    undistorted_dir = paths["undistorted_dir"]
    input_dir = paths["input_dir"]

    if not input_dir.exists():
        if not undistorted_dir.exists():
            raise FileNotFoundError(f"Missing undistorted directory: {undistorted_dir}")
        undistorted_dir.rename(input_dir)

    if undistorted_dir.exists():
        raise FileExistsError(
            f"Expected to write a new undistorted folder, but it already exists: {undistorted_dir}"
        )


def transfer_image(source_image_path: Path, target_image_path: Path, overwrite_old_data: bool):
    if not source_image_path.exists():
        raise FileNotFoundError(f"Missing undistorted image: {source_image_path}")

    if source_image_path.resolve() == target_image_path.resolve():
        return

    if target_image_path.exists():
        raise FileExistsError(f"Target image already exists: {target_image_path}")

    if overwrite_old_data:
        source_image_path.rename(target_image_path)
    else:
        shutil.copy2(source_image_path, target_image_path)


def rename_undistorted_images_and_update_reconstruction(
    scene_dir: Path,
    scene_name: str,
    overwrite_old_data: bool,
):
    paths = get_undistorted_paths(scene_dir, overwrite_old_data)
    prepare_input_output_layout(paths, overwrite_old_data)
    name_map = build_undistorted_to_original_name_map(
        scene_dir,
        scene_name,
        paths["input_images_dir"],
    )

    cameras, images, points3D = read_model(paths["input_sparse_dir"], ext=".txt")

    prepare_output_images_dir(paths, overwrite_old_data)

    updated_images = {}
    renamed_names = []
    for image_id, image in images.items():
        if image.name not in name_map:
            raise KeyError(
                f"Could not find '{image.name}' in the undistorted imagelist for scene '{scene_name}'."
            )

        new_name = name_map[image.name]
        source_image_path = paths["input_images_dir"] / image.name
        target_image_path = paths["output_images_dir"] / new_name

        transfer_image(source_image_path, target_image_path, overwrite_old_data)

        updated_images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=new_name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )
        renamed_names.append(new_name)

    imagelist_path = paths["output_images_dir"] / f"{scene_name}_undistorted.imagelist"
    imagelist_path.write_text("\n".join(renamed_names) + "\n", encoding="utf-8")

    return cameras, updated_images, points3D, paths


def deduplicate_cameras(cameras, images):
    intrinsics_to_camera_id = {}
    updated_cameras = {}
    old_to_new_camera_id = {}

    for old_camera_id in sorted(cameras):
        camera = cameras[old_camera_id]
        intrinsics_key = (
            camera.model,
            camera.width,
            camera.height,
            tuple(np.round(np.asarray(camera.params, dtype=np.float64), decimals=12)),
        )

        if intrinsics_key not in intrinsics_to_camera_id:
            new_camera_id = len(intrinsics_to_camera_id) + 1
            intrinsics_to_camera_id[intrinsics_key] = new_camera_id
            updated_cameras[new_camera_id] = Camera(
                id=new_camera_id,
                model=camera.model,
                width=camera.width,
                height=camera.height,
                params=np.asarray(camera.params, dtype=np.float64),
            )

        old_to_new_camera_id[old_camera_id] = intrinsics_to_camera_id[intrinsics_key]

    updated_images = {}
    for image_id, image in images.items():
        updated_images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=old_to_new_camera_id[image.camera_id],
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )

    return updated_cameras, updated_images


def write_processed_reconstruction(cameras, images, points3D, paths, overwrite_old_data: bool):
    output_sparse_dir = paths["output_sparse_dir"]
    if output_sparse_dir.exists() and not overwrite_old_data:
        shutil.rmtree(output_sparse_dir)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, points3D, output_sparse_dir, ext=".txt")


def process_scene(root: Path, scene_name: str, overwrite_old_data: bool):
    scene_dir = root / scene_name
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    cameras, images, points3D, paths = rename_undistorted_images_and_update_reconstruction(
        scene_dir=scene_dir,
        scene_name=scene_name,
        overwrite_old_data=overwrite_old_data,
    )
    cameras, images = deduplicate_cameras(cameras, images)
    write_processed_reconstruction(
        cameras=cameras,
        images=images,
        points3D=points3D,
        paths=paths,
        overwrite_old_data=overwrite_old_data,
    )

    print(
        f"Processed {scene_name}: {len(images)} images, "
        f"{len(cameras)} unique cameras -> {paths['output_dir']}"
    )


if __name__ == "__main__":
    args = parse_args()
    for scene_name in args.scene_names:
        process_scene(
            root=args.root,
            scene_name=scene_name,
            overwrite_old_data=args.overwrite_old_data,
        )
