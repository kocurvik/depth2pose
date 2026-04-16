from pathlib import Path
import numpy as np
from PIL import Image # pip install Pillow if not already there
import io
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import Union, Tuple, IO
import shutil

import sys
sys.path.insert(0, "./")
from datasets.colmap_utils import Camera, Point3D, write_model, rotmat2qvec
from datasets.colmap_utils import Image as CImage


# this is from https://github.com/bytedance/particle-sfm/ file particle-sfm/evaluation_evo/sintel_io.py
TAG_FLOAT = 202021.25

def compare_depths(depth1, depth2):
    # function to compare depths from different sources
    nan_mask_1 = np.isnan(depth)
    nan_mask_2 = np.isnan(depth_cached)

    num_nan_1 = np.sum(nan_mask_1)
    num_nan_2 = np.sum(nan_mask_2)

    valid_mask = ~nan_mask_1 & ~nan_mask_2

    diff = np.abs(depth - depth_cached)

    max_diff = np.max(diff[valid_mask])
    mean_diff = np.mean(diff[valid_mask])

    return [num_nan_1, num_nan_2], max_diff, mean_diff


# this function is copied from https://github.com/microsoft/MoGe/ file moge/utils/io.py
def read_depth_sintel(path: Union[str, os.PathLike, IO]) -> Tuple[np.ndarray, float]:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    unit = float(pil_image.info.get('unit')) if 'unit' in pil_image.info else None
    depth = np.array(pil_image)
    mask_nan, mask_inf = depth == 0, depth == 65535
    depth = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    return depth, unit

def depth_read_cached(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth


# this function is copied from https://github.com/bytedance/particle-sfm/ file particle-sfm/evaluation_evo/sintel_io.py
# also included in the readme of the cached collected folder
def cam_read_sintel(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N






if __name__ == "__main__":
    # in pose_basedir there are dirs for each scene and each of them contain files filename.cam
    pose_basedir = Path("/mnt/data/gg/benchmarks_original/sintel_all/sintel_pose_depth_cached/training/camdata_left/")
    
    # in depth_basedir there are dirs for each scene and each scene contains a folder filename. Each file folder contains depth.png and image.jpg, plus meta.json (intrinsics)
    depth_basedir = Path("/mnt/data/gg/benchmarks_original/sintel_all/sintel_depth_moge/Sintel/")

    out_root = Path("/mnt/data/gg/benchmarks/sintel")
       
    # I should also probably try to read the depth and intrinsics here and compare if they are the same as you get from MoGe. At least intrinsics

    # the 14 scenes used for pose evaluation (9 excluded)
    scenes_to_use = ['alley_2', 'ambush_4', 'ambush_5', 'ambush_6', 'cave_2', 'cave_4', 'market_2', 'market_5', 'market_6', 'shaman_3', 'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']
    # image dimension are the same for all images
    h, w = 436, 1024
    # go through all 14 scenes 
    for i, scene in enumerate(scenes_to_use):

        # set path for outputs
        scene_out_root = out_root / scene
        scene_out_root.mkdir(parents=True, exist_ok=True)
        save_model_path = scene_out_root / "colmap_gt"
        save_model_path.mkdir(parents=True, exist_ok=True)     
        save_image_folder = scene_out_root / "images"
        save_depth_folder = scene_out_root / "depths_gt"  
        save_image_folder.mkdir(parents=True, exist_ok=True)
        save_depth_folder.mkdir(parents=True, exist_ok=True)

        #iterate over all frames for scene
        framenames = sorted([p.name for p in (depth_basedir/scene).iterdir() if p.is_dir()])

        # to keep track of cameras
        intrinsics_to_id = {}
        next_cam_id = 0

        # vars for all cameras, images and 3D point in the scene, for colmap
        cameras, images, points3ds = {}, {}, {}

        for i, frame in enumerate(framenames):
            # set paths for input
            cam_path = pose_basedir / scene / f"{frame}.cam"
            depth_path = depth_basedir / scene / frame / "depth.png"
            im_path = depth_basedir / scene / frame / "image.jpg"
            meta_path = depth_basedir / scene / frame / "meta.json"
            
            # read pose using cam_read
            intrinsics, extrinsics = cam_read_sintel(cam_path)
            # NOT DONE HERE
            # check if the camera already exists
            key = tuple(np.round(intrinsics, 6).flatten())
            if key not in intrinsics_to_id:
                intrinsics_to_id[key] = next_cam_id
                next_cam_id += 1
            
                cam_id = intrinsics_to_id[key]

                # check if the camera is not PINHOLE
                non_pinhole_vals = np.array([intrinsics[0,1], intrinsics[1,0], intrinsics[2,0], intrinsics[2,0]])
                eps = 1e-4
                if np.any(np.abs(non_pinhole_vals) > eps) or (np.abs(intrinsics[2,2]-1.0) > eps):
                    breakpoint()

                cameras[cam_id] = Camera(
                    id = cam_id,
                    model = "PINHOLE",
                    width = w,
                    height = h,
                    params = np.array([intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]], dtype=np.float32),
                 )
            else:
                cam_id = intrinsics_to_id[key]
            
            images[i] = CImage(
                id = i,
                qvec = rotmat2qvec(extrinsics[:3, :3]),
                tvec = extrinsics[:3, 1],
                camera_id = cam_id,
                name = f"{frame}.jpg",
                xys = np.array([]),
                point3D_ids = np.array([])
            )

            points3ds[i] = Point3D(
                id = 0,
                xyz = np.array([]),
                rgb = np.array([]),
                error = 0.0,
                image_ids = np.array([]),
                point2D_idxs = np.array([])
            )
            

            # just to check we get the same depth info
            depth_cached = depth_read_cached(Path("/mnt/data/gg/benchmarks_original/sintel_all/sintel_pose_depth_cached/training/depth/") / scene / f"{frame}.dpt")


            # read depth using read_depth
            depth, _ = read_depth_sintel(depth_path)  
            
            # compare the two depth representations and check if they differ a lot
            nbr_nans, max_diff, mean_diff = compare_depths(depth, depth_cached)
            print(f"Scene {scene}, frame {frame}, depth diffs: max: {max_diff:.4f}, mean: {mean_diff:.4f}, ratio nans in MoGe2 depth {nbr_nans[0]/(h*w):.4f}, in cached depth {nbr_nans[1]/(h*w):.4f}")
            # save the depth
            
            np.savez_compressed(
                save_depth_folder / frame, depth=depth.astype(np.float32)
            )

            # copy the image
            shutil.copyfile(im_path, save_image_folder / f"{frame}.jpg")

        # write the colmap model for the whole scene
        write_model(cameras, images, points3ds, save_model_path, ext=".txt")
       

