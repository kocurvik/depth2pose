#!/bin/bash
#SBATCH --account p1358-25-2
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 1:00:00
#SBATCH --mem 120GB
#SBATCH -o /home/kocurvik/logs/eval_mdrpbench.000.std.out
#SBATCH -e /home/kocurvik/logs/eval_mdrpbench.000.err.out

# Go to your project directory
cd /home/kocurvik/code/mdrpbench
export PYTHONPATH=/home/kocurvik/code/mdrpbench
#export PATH="/home/kocurvik/mambaforge/envs/mdrpbench/bin:$PATH"
module load git

# Run your scripts using that specific Python executable
python eval_pose.py -ss -nw 64 /home/kocurvik/data/mdrpbench/pt pt splg_2048_noresize
