#!/bin/bash
#SBATCH --account p1358-25-2
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 4:00:00
#SBATCH --mem 120GB
#SBATCH -o /home/kocurvik/logs/eval_mdrpbench.001.std.out
#SBATCH -e /home/kocurvik/logs/eval_mdrpbench.001.err.out

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Go to your project directory
cd /home/kocurvik/code/mdrpbench
export PYTHONPATH=/home/kocurvik/code/mdrpbench
module load git

SRC=/home/kocurvik/data/mdrpbench/pt
WORK=/work/$SLURM_JOB_ID/mdrpbench/pt


# Copy data to work
echo "Copying data to /work..."
mkdir -p "$WORK"
rsync -a "$SRC/" "$WORK/"

# Run processing on /work
python eval_pose.py -ss -nw 64 WORK pt splg_2048_noresize

# Copy results back
echo "Copying results back..."
rsync -a "$WORK/" "$SRC/"
# rsync -a "$DATASET_WORK/" "$DATASET_SRC/"

echo "Done."

# Run your scripts using that specific Python executable

