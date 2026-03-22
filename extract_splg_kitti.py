import numpy as np
import os
import random
import h5py
from tqdm import tqdm
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

extractor = SuperPoint(
    max_num_keypoints=2048,
    detection_threshold=0.0,
    nms_radius=3,
).eval().cuda()

matcher = LightGlue(
    features='superpoint',
    depth_confidence=-1,
    width_confidence=-1,
    filter_threshold=0.1,
).eval().cuda()

folder = "/home/data/dataset/kitti_odo/data_odometry_gray/dataset/"
dataall = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

with h5py.File('/home/data/dataset/homography/test/kitti_all_splg.h5', 'w') as hf1, \
     h5py.File('/home/data/dataset/homography/test/kitti_gt.h5', 'w') as hf2, \
     open('/home/data/dataset/homography/test/kitti_pairs.txt', 'w') as ftxt:
    for scenename in tqdm(dataall):
        pose_path = os.path.join(folder, 'poses')
        image_path = os.path.join(folder, 'sequences', scenename, 'image_0')
        calib_path = os.path.join(folder, 'sequences', scenename, 'calib.txt')

        pose_file = os.path.join(pose_path, f"{scenename}.txt")
        poses = np.loadtxt(pose_file, dtype=np.float64)
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.shape[1] == 12:
            poses = poses.reshape(-1, 3, 4)
        elif poses.shape[1] == 16:
            poses = poses.reshape(-1, 4, 4)[:, :3, :]
        else:
            raise ValueError(f"Unexpected pose shape in {pose_file}: {poses.shape}")

        K_seq = None
        with open(calib_path, "r") as cf:
            for line in cf:
                if line.startswith("P0:") or line.startswith("P_rect_00:"):
                    vals = np.fromstring(line.split(":", 1)[1], sep=" ", dtype=np.float64)
                    if vals.size == 12:
                        K_seq = vals.reshape(3, 4)[:, :3]
                        break
        if K_seq is None:
            raise ValueError(f"Cannot parse K from calibration file: {calib_path}")

        images = [f for f in os.listdir(image_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        images.sort()
        valid_count = 0

        stem_to_name = {}
        for name in images:
            stem = name.split(".")[0]
            if stem.isdigit():
                stem_to_name[int(stem)] = name

        candidate_i = [i for i in stem_to_name.keys() if i >= 1 and (i + 10) in stem_to_name]
        random.shuffle(candidate_i)
        all_pairs = [(stem_to_name[i], stem_to_name[i + 10]) for i in candidate_i]

        for img1, img2 in all_pairs:
            img1_path = os.path.join(image_path, img1)
            img2_path = os.path.join(image_path, img2)

            image0 = load_image(img1_path).cuda()
            _, H1, W1 = image0.shape

            image1 = load_image(img2_path).cuda()
            _, H2, W2 = image1.shape

            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)

            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            matches = matches01['matches']
            points0 = feats0['keypoints'][matches[..., 0]]
            points1 = feats1['keypoints'][matches[..., 1]]

            kpts1c = points0.cpu().numpy()
            kpts2c = points1.cpu().numpy()
            if kpts1c.shape[0] < 50:
                continue

            match_positions = np.hstack([kpts1c[:, :2], kpts2c[:, :2]])

            img1_key = img1.split(".")[0]
            img2_key = img2.split(".")[0]

            img1_idx = int(img1_key)
            img2_idx = int(img2_key)
            if img1_idx >= poses.shape[0] or img2_idx >= poses.shape[0]:
                continue

            pose1 = poses[img1_idx]
            pose2 = poses[img2_idx]

            # KITTI poses are camera-to-world, convert to world-to-camera first.
            Tcw1 = np.eye(4, dtype=np.float64)
            Tcw1[:3, :] = pose1
            Tcw2 = np.eye(4, dtype=np.float64)
            Tcw2[:3, :] = pose2

            Twc1 = np.linalg.inv(Tcw1)
            Twc2 = np.linalg.inv(Tcw2)

            rot1 = Twc1[:3, :3]
            rot2 = Twc2[:3, :3]
            t1 = Twc1[:3, 3:4]
            t2 = Twc2[:3, 3:4]

            K1 = K_seq
            K2 = K_seq

            valid_count += 1
            if valid_count > 100:
                break

            r_gt = np.matmul(rot2, rot1.transpose())
            t_gt = t2 - np.matmul(r_gt, t1)
            Pgt = np.hstack((r_gt, t_gt))

            df = match_positions
            size1 = np.array([[int(W1)], [int(H1)]])
            size2 = np.array([[int(W2)], [int(H2)]])

            scene_clean = scenename.replace("_", "")
            img1_id = scene_clean + "_" + img1_key
            img2_id = scene_clean + "_" + img2_key
            pair_id = img1_id + "_" + img2_id

            K1name = "K_" + img1_id
            K2name = "K_" + img2_id
            S1name = "size_" + img1_id
            S2name = "size_" + img2_id

            ftxt.write(f"{img1_id} {img2_id}\n")

            hf1.create_dataset("corr_" + pair_id, data=df, compression='gzip', chunks=True)
            hf2.create_dataset("pose_" + pair_id, data=Pgt, compression='gzip', chunks=True)

            if K1name not in hf2.keys():
                hf2.create_dataset(K1name, data=K1.reshape(3, 3), compression='gzip', chunks=True)
            if S1name not in hf2.keys():
                hf2.create_dataset(S1name, data=size1, compression='gzip', chunks=True)
            if K2name not in hf2.keys():
                hf2.create_dataset(K2name, data=K2.reshape(3, 3), compression='gzip', chunks=True)
            if S2name not in hf2.keys():
                hf2.create_dataset(S2name, data=size2, compression='gzip', chunks=True)
