#!/bin/bash
#SBATCH --account p1358-25-2
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -G 1
#SBATCH -t 12:00:00
#SBATCH --mem 64GB
#SBATCH -o /home/kocurvik/logs/extract_mdrpbench_scannet.000.std.out
#SBATCH -e /home/kocurvik/logs/extract_mdrpbench_scannet.000.err.out

cd /home/kocurvik/code/mdrpbench
export PYTHONPATH=/home/kocurvik/code/mdrpbench
module load git

SRC=/home/kocurvik/data/mdrpbench/scannetpp
DATASET_SRC=/home/kocurvik/data/scannetpp

#WORK=$SRC
#DATASET_WORK=$DATASET_SRC

WORK=/work/$SLURM_JOB_ID/mdrpbench/scannetpp
DATASET_WORK=/work/$SLURM_JOB_ID/scannetpp

# Copy data to work
echo "Copying data to /work..."
mkdir -p "$WORK"
mkdir -p "$DATASET_WORK"
rsync -a "$SRC/" "$WORK/"
rsync -a "$DATASET_SRC/" "$DATASET_WORK/"

# Run processing on /work
python datasets/colmap.py --min_area_overlap 0.1 -mp 500 --check_images --name scannetpp "$WORK" "$DATASET_WORK"
python depth_estimators/infer_depth.py --name scannetpp "$WORK" "$DATASET_WORK"
python matchers/splg.py --name scannetpp "$WORK" "$DATASET_WORK"

# Copy results back
echo "Copying results back..."
rsync -a "$WORK/" "$SRC/"
# rsync -a "$DATASET_WORK/" "$DATASET_SRC/"

echo "Done."