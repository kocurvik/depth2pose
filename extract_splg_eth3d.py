import numpy as np
from pathlib import Path
import os
import random
from database import load_h5
from PIL import Image
import h5py
from tqdm import tqdm
import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from itertools import combinations

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


folder = "/home/data/dataset/eth3d_v2/"
dataall = ["courtyard","delivery_area","electro","facade","kicker","meadow","office","pipes","playground","relief","relief_2","terrace","terrains"] 
   
with h5py.File('/home/data/dataset/homography/test/eth3d_all_splg.h5', 'w') as hf1, \
     h5py.File('/home/data/dataset/homography/test/eth3d_gt.h5', 'w') as hf2, \
         open('/home/data/dataset/homography/test/eth3d_pairs.txt', 'w') as ftxt:
    for scenename in tqdm(dataall):  # 
        
        pose_path = "/home/data/dataset/eth3d_v2/"+scenename+"/gt/"
        
        intri_data = load_h5(pose_path + 'K.h5')
        rot_data = load_h5(pose_path + 'R.h5')
        T_data = load_h5(pose_path + 'T.h5')
        
        image_path = "/home/data/dataset/eth3d_v2/"+scenename+"/images/"
        images = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        valid_count = 0
        
        pair_set = set()
        pair_set.clear() 
        
        all_pairs = list(combinations(images, 2))  
        random.shuffle(all_pairs)  
        
        for img1, img2 in all_pairs:

            img1_path = os.path.join(image_path, img1)
            img2_path = os.path.join(image_path, img2)
        
            # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
            image0 = load_image(img1_path).cuda()
            C1, H1, W1 = image0.shape

            image1 = load_image(img2_path).cuda()
            C2, H2, W2 = image1.shape

            # Match
            feats0 = extractor.extract(image0)  
            feats1 = extractor.extract(image1)
        
            # match the features
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
            
            kpts1c = points0.cpu().numpy()
            kpts2c = points1.cpu().numpy()
            
            if kpts1c.shape[0] < 50:
                continue
        
            match_positions = np.hstack([kpts1c[:, :2], kpts2c[:, :2]])
            
            img1_key = img1.split(".")[0]
            img2_key = img2.split(".")[0]

            if img1_key not in rot_data or img2_key not in rot_data:
                continue  
            
            
            rot1 = np.array(rot_data[img1_key])
            rot2 = np.array(rot_data[img2_key])
            
            K1 = np.array(intri_data[img1_key])
            K2 = np.array(intri_data[img2_key])

            t1 = np.array(T_data[img1_key])
            t2 = np.array(T_data[img2_key])
            
            valid_count += 1
            
            if valid_count > 100:
                break

            r_gt = np.matmul(rot2,rot1.transpose())
            t_gt = t2-np.matmul(r_gt,t1)
        
            Pgt = np.hstack((r_gt,t_gt))
            
            df = match_positions
            size1 = np.array([[int(W1)],[int(H1)]])
            size2 = np.array([[int(W2)],[int(H2)]])
            
            scene_clean = scenename.replace("_", "")
            
            rela0 = scene_clean + "_" + img1.split('.')[0]
            rela1 = scene_clean + "_" + img2.split('.')[0]
            K1name = "K_"+rela0
            K2name = "K_"+rela1
            S1name = "size_"+rela0
            S2name = "size_"+rela1
            
            ftxt.write(f"{scenename}_{img1} {scenename}_{img2}\n")
            
            hf1.create_dataset("corr_"+rela0+"_"+rela1, data=df, compression='gzip', chunks=True)
            
            hf2.create_dataset("pose_"+rela0+"_"+rela1, data=Pgt, compression='gzip', chunks=True)
            
            if not K1name in hf2.keys():
                hf2.create_dataset(K1name, data=K1.reshape(3,3), compression='gzip', chunks=True)
            if not S1name in hf2.keys():
                hf2.create_dataset(S1name, data=size1, compression='gzip', chunks=True)
            if not K2name in hf2.keys():
                hf2.create_dataset(K2name, data=K2.reshape(3,3), compression='gzip', chunks=True)
            if not S2name in hf2.keys():
                hf2.create_dataset(S2name, data=size2, compression='gzip', chunks=True)
                
                