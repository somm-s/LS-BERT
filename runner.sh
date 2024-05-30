#!/bin/bash

#SBATCH --job-name=LS-BERT-JUP
#SBATCH --output=out/%x-%j.out
#SBATCH --error=err/%x-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH --partition=nodes
#SBATCH --gres=gpu:titanv:1
#SBATCH --chdir=/cluster/raid/home/sosi/repos/LS-BERT/

# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Activate environment
ls
source ~/repos/LS-BERT/myenv/bin/activate
jupyter notebook --no-browser --port=9998
deactivate
