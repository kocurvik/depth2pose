#!/bin/bash
#SBATCH --account p1358-25-2
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -G 1
#SBATCH -t 4:00:00
#SBATCH --mem 64GB
#SBATCH -o /home/kocurvik/logs/eval_depth.000.std.out
#SBATCH -e /home/kocurvik/logs/eval_depth.000.err.out

# Go to your project directory
cd /home/kocurvik/code/mdrpbench
export PYTHONPATH=/home/kocurvik/code/mdrpbench
module load git

# Run processing on /work
python eval_depth.py /projects/p1358-25-2/mdrpbench/all_results ./datasets/devana_benchmarks.json --device cuda

echo "Done."


