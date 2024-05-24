#!/bin/bash

#SBATCH --job-name=NIDS
#SBATCH --output=out/%x-%j.out
#SBATCH --error=err/%x-%j.out
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=nodes
#SBATCH --gres=gpu:titanv:0
#SBATCH --chdir=/cluster/raid/home/sosi/repos/Bert/

# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# free -h
# lscpu

# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Activate environment
source ~/repos/Bert/.venv/bin/activate
lsof -ti :12346 | xargs kill -9
ssh -p 6722 -fN -R 12346:localhost:12346 lsansible@jump.lockedshields.ch
python ls_runner.py #pipeline-script.py

# kill the ssh tunnel
lsof -ti :12346 | xargs kill -9

deactivate