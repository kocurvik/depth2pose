#!/bin/bash
#SBATCH --account p1358-25-2
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -G 1
#SBATCH -t 4:00:00
#SBATCH --mem 64GB
#SBATCH -o /home/kocurvik/logs/extract_mdrpbench.000.std.out
#SBATCH -e /home/kocurvik/logs/extract_mdrpbench.000.err.out

cd /home/kocurvik/code/mdrpbench
export PYTHONPATH=/home/kocurvik/code/mdrpbench
module load git

SRC=/home/kocurvik/data/mdrpbench/eth3d_800
DATASET_SRC=/home/kocurvik/data/ETH3D/ETH3D_depth

#WORK=$SRC
#DATASET_WORK=$DATASET_SRC

WORK=/work/$SLURM_JOB_ID/mdrpbench/eth3d
DATASET_WORK=/work/$SLURM_JOB_ID/ETH3D_depth

# Copy data to work
echo "Copying data to /work..."
mkdir -p "$WORK"
mkdir -p "$DATASET_WORK"
rsync -a "$SRC/" "$WORK/"
rsync -a "$DATASET_SRC/" "$DATASET_WORK/"

# Run processing on /work
#python datasets/colmap.py --min_area_overlap 0.025 --check_images --name eth3d "$WORK" "$DATASET_WORK"
#python depth_estimators/infer_depth.py --name eth3d "$WORK" "$DATASET_WORK"
#python matchers/splg.py --name eth3d "$WORK" "$DATASET_WORK"

python datasets/colmap.py --min_area_overlap 0.025 --check_images --max_resolution 800 --name eth3d_800 "$WORK" "$DATASET_WORK"
python depth_estimators/infer_depth.py --name eth3d_800 "$WORK" "$DATASET_WORK"
python matchers/splg.py --name eth3d_800 "$WORK" "$DATASET_WORK"

# Copy results back
echo "Copying results back..."
rsync -a "$WORK/" "$SRC/"
# rsync -a "$DATASET_WORK/" "$DATASET_SRC/"

echo "Done."