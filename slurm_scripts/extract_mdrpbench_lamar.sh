#!/bin/bash
#SBATCH --account p1358-25-2
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -G 1
#SBATCH -t 2:00:00
#SBATCH --mem 64GB
#SBATCH -o /home/kocurvik/logs/extract_mdrpbench_lamar.000.std.out
#SBATCH -e /home/kocurvik/logs/extract_mdrpbench_lamar.000.err.out

cd /home/kocurvik/code/mdrpbench
export PYTHONPATH=/home/kocurvik/code/mdrpbench
module load git

PROCESSED=/projects/p1358-25-2/mdrpbench/lamar
DATASET=/projects/p1358-25-2/data/lamar


# Run processing on /work
python datasets/colmap.py --min_area_overlap 0.1 --check_images -mp 2500 --name lamar "$PROCESSED" "$DATASET"
python matchers/splg.py --name lamar "$PROCESSED" "$DATASET"
# python depth_estimators/infer_depth.py --name lamar "$WORK" "$DATASET_WORK"
python spawn_infer_depth.py --work_dir --name lamar "$PROCESSED" "$DATASET"

echo "Done."