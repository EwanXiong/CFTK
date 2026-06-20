#!/bin/bash
#SBATCH --job-name=cftk_test
#SBATCH --output=log/report.out
#SBATCH --error=log/report.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --partition=standard
#SBATCH --account=weil21_lab

source ~/.bashrc
conda activate twist

python -m sphinx -b html . html
