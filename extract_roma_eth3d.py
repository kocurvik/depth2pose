from PIL import Image
import torch
import cv2
from romatch import roma_outdoor
from pathlib import Path
import os
import numpy as np
from database import load_h5
import math
import sqlite3
import itertools
import h5py
import glob
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

roma_model = roma_outdoor(device=device)

with open("/home/data/dataset/homography/test/eth3d_pairs.txt", "r") as f, h5py.File('/home/data/dataset/homography/test/eth3d_all_roma.h5', 'w') as hf1:
    for line in f:
        
        img1, img2 = line.strip().split()
    
        scenename = img1.split('_DSC_')[0]
    
    
        image_path = "/home/data/dataset/eth3d_v2/"+scenename+"/images/"
    
        image_name1 = 'DSC_' + img1.split('_DSC_')[1]
        image_name2 = 'DSC_' + img2.split('_DSC_')[1]

        img1_path = os.path.join(image_path, image_name1)
        img2_path = os.path.join(image_path, image_name2)
        
        W_A, H_A = Image.open(img1_path).size
        W_B, H_B = Image.open(img2_path).size
        
        # Match
        warp, certainty = roma_model.match(img1_path, img2_path, device=device)
        matches, certainty = roma_model.sample(warp, certainty, num=2048)
        kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)     
        
        kpts1c = kpts1.cpu().numpy()
        kpts2c = kpts2.cpu().numpy()
        
        match_positions = np.hstack([kpts1c[:, :2], kpts2c[:, :2]])
        
        df = match_positions
        
        scene_clean = scenename.replace("_", "")
        
        rela0 = scene_clean + "_" + image_name1.split('.')[0]
        rela1 = scene_clean + "_" + image_name2.split('.')[0]
        
        hf1.create_dataset("corr_"+rela0+"_"+rela1, data=df, compression='gzip', chunks=True)

