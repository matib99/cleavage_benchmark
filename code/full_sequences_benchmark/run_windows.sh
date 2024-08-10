#!/bin/bash

#SBATCH --job-name=csp_predict_windows
#SBATCH --partition=common
#SBATCH --qos=1gpu2d
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/matib99/job_outputs/windows/out.txt
#SBATCH --error=/home/matib99/job_outputs/windows/err.txt

source /home/matib99/miniconda3/etc/profile.d/conda.sh
conda init
conda activate cleavage_benchmark

python ./code/full_sequences_benchmark/windows_predict.py