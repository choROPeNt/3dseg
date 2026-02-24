#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=6:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=8                               # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --mem-per-cpu=48G                             # memory per CPU core
#SBATCH --gres=gpu:1                                    # number of gpus
#SBATCH -J "torch_3dseg"                           # job name
#SBATCH --output=slurm_out/3dseg_BIIAX_run01-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_biiax
##################################################################
##################################################################

ml release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1

source .venv_torch/bin/activate

train_config=configs/NEAPEL/model_64x64x64_bin/config_train.yml
pred_config=configs/NEAPEL/model_64x64x64_bin/config_pred.yml

python ./scripts/train.py --config $train_config

python ./scripts/predict.py --config $pred_config

exit 0