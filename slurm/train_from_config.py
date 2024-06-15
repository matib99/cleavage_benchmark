import random
import string
import sys
import os

from pathlib import Path




alphabet = string.ascii_lowercase + string.digits

job_id = ''.join(random.choices(alphabet, k=8))
config_file = os.path.expanduser(sys.argv[1])
time_h = int(sys.argv[2]) if len(sys.argv) > 2 else 24

if time_h > 23:
    time_h = f"{time_h // 24}-{(time_h % 24):02d}:00:00"
else:
    time_h = f"{time_h:02d}:00:00"

config_file_name = config_file.split('/')[-1]
job_name = f"{config_file_name.split('.')[0]}_{job_id}"

directory = os.path.expanduser(f"~/job_outputs/{job_name}")
os.system(f"mkdir -p {directory}/res/")

run_sh_path = f"{directory}/run.sh"
os.system(f"touch {run_sh_path}")

slurm_script = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition=common
#SBATCH --qos=1gpu2d
#SBATCH --gres=gpu:1
#SBATCH --time={time_h}
#SBATCH --output={directory}/out.txt
#SBATCH --error={directory}/err.txt

source ~/.bashrc
conda init
conda activate cleavage_benchmark

python ./code/test_slurm.py @{config_file} --saving_path {directory}/res/
"""
# script_file = Path(f"{directory}/run.sh")
# script_file.touch(exist_ok=True) 
# run bash script
with open(f"{run_sh_path}", "w+") as f:
    f.write(slurm_script)
os.system(f"chmod +x {run_sh_path}")
os.system(f"sbatch {run_sh_path}")

# python ../code/run_train.py @{config_file} --saving_path {directory}/res/