import shutil
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, "./")
from datasets.colmap_utils import Camera, Point3D, write_model, rotmat2qvec
from datasets.colmap_utils import Image as CImage


DEPTH_SCALE = 1000
DEPTH_MIN = 0
DEPTH_MAX = 5


def read_depth(depth_file):
    depth = Image.open(depth_file)
    depth = np.array(depth, dtype=np.uint16)
    print(depth.shape, depth.min(), depth.max())
    return depth / DEPTH_SCALE

def read_pose(datapath: Path):
    poses = np.genfromtxt(datapath / "cam_poses.txt")
    poses = poses[:, 1:].reshape(-1, 4, 4)
    poses = np.linalg.inv(poses)
    return poses


if __name__ == "__main__":
    root = Path("/mnt/data/gg/benchmarks_original/wildrgbd")
    out_root = Path("/mnt/data/gg/benchmarks/wildrgbd")

    if out_root.exists():
        shutil.rmtree(out_root)

    scenes = sorted(list(root.glob("*")))
    scenes = [scene.stem for scene in scenes if scene.is_dir()]

    for scene in scenes:
        print(scene)
        poses = read_pose(root / scene)

        with open(root / scene / "metadata") as f:
            meta = json.load(f)

        K = np.array(meta['K']).reshape(3, 3).T
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        w, h = meta['w'], meta['h']

        save_model_path = out_root / scene / "colmap_gt"
        save_image_folder = out_root / scene / "images"
        save_depth_folder = out_root / scene / "depths_gt"
        save_model_path.mkdir(parents=True, exist_ok=True)
        save_image_folder.mkdir(parents=True, exist_ok=True)
        save_depth_folder.mkdir(parents=True, exist_ok=True)

        cameras, images, points3ds = {}, {}, {}
        cameras[0] = Camera(
            id=0,
            model="PINHOLE",
            width=w,
            height=h,
            params=np.array([fx, fy, cx, cy]),
        )

        for i in tqdm(range(poses.shape[0])):
            filename = f"{i:0>5d}"
            extrinsic = poses[i]

            # copy image file
            shutil.copyfile(
                root / scene / "rgb" / f"{filename}.png",
                save_image_folder / f"{filename}.png"
            )

            depth = read_depth(root / scene / "depth" / f"{filename}.png")
            np.savez_compressed(save_depth_folder / filename, depth=depth.astype(np.float32))

            images[i] = CImage(
                id=i,
                qvec=rotmat2qvec(extrinsic[:3, :3]),
                tvec=extrinsic[:3, -1],
                camera_id=i,
                name=f"{filename}.png",
                xys=np.array([]),
                point3D_ids=np.array([])
            )
            points3ds[i] = Point3D(
                id=i,
                xyz=np.array([]),
                rgb=np.array([]),
                error=0.0,
                image_ids=np.array([]),
                point2D_idxs=np.array([])
            )
            # break

        # save a new colmap model
        write_model(cameras, images, points3ds, save_model_path, ext=".txt")

        # break
