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

SRC=/home/kocurvik/data/mdrpbench/pt
DATASET_SRC=/home/kocurvik/data/pt_test

WORK=$SRC
DATASET_WORK=$DATASET_SRC

#WORK=/work/$SLURM_JOB_ID/mdrpbench/pt
#DATASET_WORK=/work/$SLURM_JOB_ID/pt_test

# Copy data to work
#echo "Copying data to /work..."
#mkdir -p "$WORK"
#mkdir -p "$DATASET_WORK"
#rsync -a "$SRC/" "$WORK/"
#rsync -a "$DATASET_SRC/" "$DATASET_WORK/"

# Run processing on /work
python datasets/colmap.py --check_images -mp 500 --name pt "$WORK" "$DATASET_WORK"
python depth_estimators/infer_depth.py --name pt "$WORK" "$DATASET_WORK"
python matchers/splg.py --name pt "$WORK" "$DATASET_WORK"

# Copy results back
#echo "Copying results back..."
#rsync -a "$WORK/" "$SRC/"
# rsync -a "$DATASET_WORK/" "$DATASET_SRC/"

echo "Done."