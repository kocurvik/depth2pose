import pycolmap
from pathlib import Path
import imageio
import cv2
import numpy as np
import shutil

DEPTH_SCALE = 1000.

def read_depth(path: Path) -> np.ndarray:
    # Much faster than PIL.Image for high-res images
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH) / DEPTH_SCALE
    return depth

def keep_image(name, keep_prefixes, keep_cam):
    return (
        any(name.startswith(prefix) for prefix in keep_prefixes)
        and keep_cam in name
    )

def subsample_colmap_txt(sparse_dir_orig, sparse_dir_new, im_parent_dir, keep_dirs_prefixes, keep_cam, change_name = False): 
    # subsamples the colmap reconstruction in sparse_dir_orig to only contain images from the image directiories in im_parent_dir starting with keep_dirs_prefix.
    # then saves the new reconstruction to sparse_dir_new
    
    sparse_dir_orig = Path(sparse_dir_orig)
    sparse_dir_new = Path(sparse_dir_new)
    sparse_dir_new.mkdir(exist_ok=True)
    
    kept_images = {}
    kept_images_newnames = {}
    kept_image_ids = set()

    with open(sparse_dir_orig / "images.txt") as f, open(sparse_dir_new / "images.txt", "w") as out:
        lines = f.readlines()
        i = 0

        # filter the images
        while i < len(lines):
            line = lines[i]

            # if comment, just copy
            if line.startswith("#"):
                out.write(line)
                i += 1
                continue

            header = line.strip().split()
            image_id = int(header[0])
            name = header[-1]
                
            points_line = lines[i + 1]

            # check if images is in one of the correct directories
            if keep_image(name, keep_dirs_prefixes, keep_cam):
                # keep not of saved image
                kept_images[image_id] = header
                kept_image_ids.add(image_id)

                # write into new image txt file, first image info, then 2D points
                if change_name:
                    folder, imname = name.split('/')
                    newname = folder + imname
                    kept_images_newnames[image_id] = newname
                    out.write(' '.join(header[:-1]) + ' ' + newname +  "\n")
                else: 
                    out.write(lines[i])
                
                out.write(points_line)

            
            i += 2

        # filter the 3D points

        kept_point_ids = set()

        with open(sparse_dir_orig / "points3D.txt") as f, open(sparse_dir_new / "points3D.txt", "w") as out:
            for line in f:
                # just copy comments
                if line.startswith("#"):
                    out.write(line)
                    continue

                elems = line.strip().split()
                point_id = int(elems[0])

                track = elems[8:]
                new_track = []

                for j in range(0, len(track), 2):
                    img_id = int(track[j])
                    pt2d_id = track[j+1]

                    if img_id in kept_image_ids:
                        # save image id to the new track
                        new_track.extend([str(img_id), pt2d_id])

                if len(new_track) >= 4: # two observations
                    elems[8:] = new_track
                    out.write(" ".join(elems) + "\n")
                    kept_point_ids.add(point_id)

            # filter cameras
            used_camera_ids = set()

            for img in kept_images.values():
                # camera id is 9th value
                used_camera_ids.add(int(img[8])) 

            with open(sparse_dir_orig / "cameras.txt") as f, open(sparse_dir_new / "cameras.txt", "w") as out:
                for line in f:
                    if line.startswith("#"):
                        out.write(line)
                        continue
                    
                    cam_id = int(line.split()[0])
                    if cam_id in used_camera_ids:
                        out.write(line)

            print("Pruned reconstruction saved in txt format in " + str(sparse_dir_new))

    return kept_image_ids, kept_images, kept_images_newnames



if __name__ == "__main__":
    # subsample the colmap reconstruction to only contain images of certain directories/sessions

    scenes = ["CAB", "HGE", "LIN"]
    sparse_root = Path("/mnt/data/gg/benchmarks_original/lamar/colmap/")
    out_root = Path("/mnt/data/gg/benchmarks/lamar/")
    change_name = True

    for scene in scenes:   
        print(f"Working on scene {scene}")
        sparse_dir_orig = sparse_root / scene / "sparse_txt"
        im_parent_dir = sparse_root / scene / "images"
        depth_parent_dir = sparse_root / scene / "navvis_depth"

        save_model_path = out_root / scene / "colmap_gt"
        save_image_folder = out_root / scene / "images"
        save_depth_folder = out_root / scene / "depths_gt"
        save_model_path.mkdir(parents=True, exist_ok=True)
        save_image_folder.mkdir(parents=True, exist_ok=True)
        save_depth_folder.mkdir(parents=True, exist_ok=True)

        # choose the folder/data type
        keep_im_dirs_prefix = ["navvis"]
        # for the devices that have several cameras, choose which one to use
        keep_cam = "cam1" # cam1 and cam3 are forward or backward, while cam0 and cam2 seems to be to the sides, based on colmap recon.

        # subsample and save colmap reconstruction
        kept_images_ids, kept_images, kept_images_newnames = subsample_colmap_txt(sparse_dir_orig, save_model_path, im_parent_dir, keep_im_dirs_prefix, keep_cam, change_name)

        for image_id in kept_images_ids:
            header = kept_images[image_id]
            name = header[-1]
            newname = kept_images_newnames[image_id]

            im_path = im_parent_dir / name
            depth_path = (depth_parent_dir / name).with_suffix('.png')

            # read and save the depth (contains rescaling)
            depth = read_depth(depth_path) 
            np.savez_compressed(
                save_depth_folder / f"{Path(newname).stem}", depth=depth.astype(np.float32)
            )
            #print("Depth saved in np format in " + str(save_depth_folder / Path(newname).stem))
            # copy the image
            shutil.copyfile(im_parent_dir / name, save_image_folder / newname)
            #print("Image saved in " + str(save_image_folder) + newname)

        print(f"Finished preprocessing and saving scene {scene}")
