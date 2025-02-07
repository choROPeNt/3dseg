#!/bin/bash -l

ml release/23.04 GCC/12.2.0 Python/3.10.8 OpenMPI/4.1.4 CUDA/11.8.0

source /home/dchristi/projects_alpha/3dseg/.venv_3dseg/bin/activate

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