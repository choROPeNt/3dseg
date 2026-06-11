#!/bin/bash -l

ml release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1

source /home/dchristi/projects_alpha/3dseg/.venv_torch/bin/activate

#FIXED Arguments / pathes are absolute!
BASE_CONFIG=/home/dchristi/projects_alpha/3dseg/configs/OmniOpt/train_config.yml
DIR=/home/dchristi/projects_alpha/3dseg/configs/OmniOpt_01
CHECKPOINT_DIR=/data/horse/ws/dchristi-3dseg/checkpoints/PoreDetection/

echo $@
# Load your script. $@ is all the parameters that are given to this run.sh file.
python /home/dchristi/projects_alpha/3dseg/scripts/run_omniopt.py \
  --config=$BASE_CONFIG \
  --dir=$DIR \
  --checkpoint_dir=$CHECKPOINT_DIR \
  $@