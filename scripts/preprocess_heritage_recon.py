import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import pycolmap
import yaml
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render point-cloud depth maps for Heritage Recon scenes."
    )
    parser.add_argument(
        "--reconstruction_path",
        type=Path,
        required=True,
        help="Path to the COLMAP sparse model directory, e.g. scene/dense/sparse.",
    )
    parser.add_argument(
        "--ply_path",
        type=Path,
        required=True,
        help="Path to the object point-cloud .ply file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where depth maps and optional visualizations will be written.",
    )
    parser.add_argument(
        "--scene_config",
        type=Path,
        default=None,
        help="Path to config.yaml containing sfm2gt. Defaults to <scene_root>/config.yaml.",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=None,
        help="Directory containing the undistorted reconstruction images for overlays.",
    )
    parser.add_argument(
        "--ply_frame",
        choices=["gt", "sfm"],
        default="gt",
        help="Coordinate frame of the input point cloud.",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=1,
        help="Pixel splat radius for point rendering.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional image limit for debugging.",
    )
    parser.add_argument(
        "--image_list",
        type=Path,
        default=None,
        help="Optional text file with one reconstruction image name per line.",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Also save grayscale depth previews and RGB overlays.",
    )
    return parser.parse_args()


def infer_scene_config_path(reconstruction_path: Path) -> Path:
    reconstruction_path = reconstruction_path.resolve()
    return reconstruction_path.parents[1] / "config.yaml"


def infer_images_dir(reconstruction_path: Path) -> Path | None:
    reconstruction_path = reconstruction_path.resolve()

    dense_images_dir = reconstruction_path.parent / "images"
    if dense_images_dir.is_dir():
        return dense_images_dir

    scene_images_dir = reconstruction_path.parents[1] / "images"
    if scene_images_dir.is_dir():
        return scene_images_dir

    return None


def load_sfm2gt(scene_config_path: Path) -> np.ndarray:
    with open(scene_config_path, "r", encoding="utf-8") as handle:
        scene_config = yaml.load(handle, Loader=yaml.FullLoader)

    sfm2gt = np.asarray(scene_config["sfm2gt"], dtype=np.float64)
    if sfm2gt.shape != (4, 4):
        raise ValueError(f"Expected sfm2gt to have shape (4, 4), got {sfm2gt.shape}")
    return sfm2gt


def make_intrinsics_matrix(camera) -> np.ndarray:
    model = str(camera.model).split(".")[-1]
    params = camera.params

    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")

    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def image_world_to_camera_sfm(image) -> np.ndarray:
    cam_from_world = image.cam_from_world()

    if hasattr(cam_from_world, "matrix"):
        matrix = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        if matrix.shape == (3, 4):
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :4] = matrix
            return world_to_camera
        if matrix.shape == (4, 4):
            return matrix

    if hasattr(cam_from_world.rotation, "matrix"):
        rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    else:
        raise ValueError("Could not extract camera rotation from pycolmap image transform.")

    translation = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return world_to_camera


def load_point_cloud(ply_path: Path) -> np.ndarray:
    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"No points found in {ply_path}")
    return points


def project_points_to_depth(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    width: int,
    height: int,
    point_radius: int,
) -> np.ndarray:
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    points_camera = (rotation @ points_world.T + translation[:, None]).T

    z = points_camera[:, 2]
    valid = z > 1e-8
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.float32)

    points_camera = points_camera[valid]
    z = points_camera[:, 2].astype(np.float32)

    u = intrinsic[0, 0] * (points_camera[:, 0] / z) + intrinsic[0, 2]
    v = intrinsic[1, 1] * (points_camera[:, 1] / z) + intrinsic[1, 2]
    u = np.rint(u).astype(np.int32)
    v = np.rint(v).astype(np.int32)

    depth = np.full((height, width), np.inf, dtype=np.float32)
    flat_depth = depth.reshape(-1)

    for du in range(-point_radius, point_radius + 1):
        for dv in range(-point_radius, point_radius + 1):
            uu = u + du
            vv = v + dv
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not np.any(inside):
                continue

            flat_idx = vv[inside] * width + uu[inside]
            np.minimum.at(flat_depth, flat_idx, z[inside])

    depth[np.isinf(depth)] = 0.0
    return depth


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    preview = np.zeros_like(depth, dtype=np.float32)
    mask = depth > 0
    if np.any(mask):
        valid = depth[mask]
        near = float(valid.min())
        far = float(valid.max())
        if far > near:
            preview[mask] = (valid - near) / (far - near)
        else:
            preview[mask] = 1.0
    return np.round(preview * 255.0).astype(np.uint8)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.mod(h, 1.0)
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    rgb = np.zeros(h.shape + (3,), dtype=np.float32)
    rgb[i == 0] = np.stack([v[i == 0], t[i == 0], p[i == 0]], axis=-1)
    rgb[i == 1] = np.stack([q[i == 1], v[i == 1], p[i == 1]], axis=-1)
    rgb[i == 2] = np.stack([p[i == 2], v[i == 2], t[i == 2]], axis=-1)
    rgb[i == 3] = np.stack([p[i == 3], q[i == 3], v[i == 3]], axis=-1)
    rgb[i == 4] = np.stack([t[i == 4], p[i == 4], v[i == 4]], axis=-1)
    rgb[i == 5] = np.stack([v[i == 5], p[i == 5], q[i == 5]], axis=-1)
    return rgb


def depth_to_color(depth: np.ndarray) -> np.ndarray:
    color = np.zeros(depth.shape + (3,), dtype=np.uint8)
    mask = depth > 0
    if not np.any(mask):
        return color

    valid = depth[mask].astype(np.float32)
    near = float(valid.min())
    far = float(valid.max())
    if far > near:
        norm = (valid - near) / (far - near)
    else:
        norm = np.zeros_like(valid)

    hue = (0.85 - 0.85 * norm) % 1.0
    rgb = hsv_to_rgb(hue, np.full_like(hue, 0.85), np.full_like(hue, 1.0))
    color[mask] = np.round(rgb * 255.0).astype(np.uint8)
    return color


def load_image_for_overlay(images_dir: Path | None, image_name: str) -> np.ndarray | None:
    if images_dir is None:
        return None

    image_path = images_dir / image_name
    if not image_path.is_file():
        return None

    image = imageio.imread(image_path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def save_image_and_depth_outputs(
    depth: np.ndarray,
    output_dir: Path,
    image_name: str,
    save_visualization: bool,
    rgb_image: np.ndarray | None,
    rgb_path: Path,
):
    image_path = Path(image_name)

    # copy the rgb image
    save_im_path = output_dir / "images" / image_name
    save_im_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rgb_path / image_name, save_im_path)

    depth_npz_path = output_dir / "depths_gt" / image_path.with_suffix(".npz")
    depth_npz_path.parent.mkdir(parents=True, exist_ok=True)
    #np.save(depth_npy_path, depth.astype(np.float32))
    np.savez_compressed(depth_npz_path, depth=depth.astype(np.float32))

    if not save_visualization:
        return

    depth_vis_path = output_dir / "depth_vis" / image_path.with_suffix(".png")
    depth_vis_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(depth_vis_path, depth_to_uint8(depth))

    if rgb_image is None:
        return

    depth_color = depth_to_color(depth)
    mask = depth > 0
    overlay = rgb_image.astype(np.float32).copy()
    overlay[mask] = 0.55 * rgb_image[mask].astype(np.float32) + 0.45 * depth_color[mask].astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    overlay_path = output_dir / "depth_vis" / image_path.with_name(f"{image_path.stem}_overlay.png")
    imageio.imwrite(overlay_path, overlay)


def get_selected_images(reconstruction, image_list_path: Path | None, max_images: int | None):
    selected_names = None
    if image_list_path is not None:
        with open(image_list_path, "r", encoding="utf-8") as handle:
            selected_names = {line.strip() for line in handle if line.strip()}

    images = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        if selected_names is not None and image.name not in selected_names:
            continue
        images.append(image)
        if max_images is not None and len(images) >= max_images:
            break

    return images


def main(args):
    scene_config_path = args.scene_config or infer_scene_config_path(args.reconstruction_path)
    images_dir = args.images_dir or infer_images_dir(args.reconstruction_path)

    sfm2gt = load_sfm2gt(scene_config_path)
    gt2sfm = np.linalg.inv(sfm2gt)

    reconstruction = pycolmap.Reconstruction(str(args.reconstruction_path))
    point_cloud = load_point_cloud(args.ply_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_images = get_selected_images(reconstruction, args.image_list, args.max_images)

    processed = 0
    for image in selected_images:
        camera = reconstruction.cameras[image.camera_id]
        intrinsic = make_intrinsics_matrix(camera)
        world_to_camera_sfm = image_world_to_camera_sfm(image)

        if args.ply_frame == "gt":
            world_to_camera = world_to_camera_sfm @ gt2sfm
        else:
            world_to_camera = world_to_camera_sfm

        depth = project_points_to_depth(
            points_world=point_cloud,
            intrinsic=intrinsic,
            world_to_camera=world_to_camera,
            width=int(camera.width),
            height=int(camera.height),
            point_radius=args.point_radius,
        )

        rgb_image = None
        if args.save_visualization:
            rgb_image = load_image_for_overlay(images_dir, image.name)

        save_image_and_depth_outputs(
            depth=depth,
            output_dir=args.output_dir,
            image_name=image.name,
            save_visualization=args.save_visualization,
            rgb_image=rgb_image,
            rgb_path = images_dir
        )

        processed += 1

        if processed % 25 == 0:
            print(f"Processed {processed} images")

    print(f"Finished rendering depth for {processed} images into {args.output_dir}")


if __name__ == "__main__":
    # args = parse_args()
    
    # For local debugging, you can replace the line above with:
    # args = argparse.Namespace(
    #     reconstruction_path=Path("/path/to/scene/dense/sparse"),
    #     ply_path=Path("/path/to/object.ply"),
    #     output_dir=Path("/path/to/output"),
    #     scene_config=None,
    #     images_dir=None,
    #     ply_frame="gt",
    #     point_radius=1,
    #     max_images=None,
    #     image_list=None,
    #     save_visualization=True,
    # )
    basepath = Path("/mnt/data/gg/benchmarks_original/heritage_recon/")
    base_out = Path("/mnt/data/gg/benchmarks/heritage_recon/")
    scenes = ['brandenburg_gate', 'lincoln_memorial', 'palacio_de_bellas_artes', 'pantheon_exterior']
    scenes = ['palacio_de_bellas_artes']
    for scene in scenes:
        # copy the colmap reconstruction, as it is
        shutil.copytree(basepath / scene / "dense/sparse/", base_out / scene/ "colmap_gt/", dirs_exist_ok=True)

        # create the depth image and save that and the rgb image
        args = argparse.Namespace(
            reconstruction_path=basepath / f"{scene}/dense/sparse/",
            ply_path=basepath / scene / f"{scene}.ply",
            output_dir= base_out / scene,
            scene_config= basepath / f"{scene}/config.yaml",
            images_dir=None,
            ply_frame="gt",
            point_radius=1,
            max_images=None,
            image_list=None,
            save_visualization=True,
        )
        main(args)
        
        