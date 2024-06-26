#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=5:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=8                               # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --mem-per-cpu=8G                             # memory per CPU core
#SBATCH --gres=gpu:1                                    # number of gpus
#SBATCH -J "torch_3dseg"                           # job name
#SBATCH --output=slurm_out/3dseg_BIIAX_lowres-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_autoshear
##################################################################
##################################################################

module load release/23.04 GCC/12.2.0 Python/3.10.8 OpenMPI/4.1.4 CUDA/11.8.0

source .venv_3dseg/bin/activate

# train_config=./configs/BIIAX/00-lowres-3c/train.yml
# pred_config=./configs/BIIAX/00-lowres-3c/pred.yml

train_config=./configs/pore-detection/train_config_pore.yml


python ./scripts/train.py --config $train_config

# python ./scripts/predict.py --config $pred_config

exit 0