#!/bin/bash
#SBATCH --account p1358-25-2
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -G 1
#SBATCH -t 1:00:00
#SBATCH --mem 64GB
#SBATCH -o /home/kocurvik/logs/extract_mdrpbench.000.std.out
#SBATCH -e /home/kocurvik/logs/extract_mdrpbench.000.err.out

# Go to your project directory
cd /home/kocurvik/code/mdrpbench
export PYTHONPATH=/home/kocurvik/code/mdrpbench
module load git

# Run your scripts using that specific Python executable
python datasets/colmap.py --check_images -mp 500 --name pt /home/kocurvik/data/mdrpbench/pt /home/kocurvik/data/pt_test
python depth_estimators/infer_depth.py --name pt /home/kocurvik/data/mdrpbench/pt /home/kocurvik/data/pt_test
python matchers/splg.py --name pt /home/kocurvik/data/mdrpbench/pt /home/kocurvik/data/pt_test
