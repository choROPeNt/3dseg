#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=8:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=24                               # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --mem-per-cpu=16G                             # memory per CPU core
#SBATCH --gres=gpu:4                                    # number of gpus
#SBATCH -J "3dseg-torch_predict"                           # job name
#SBATCH --output=/lustre/ssd/ws/dchristi-3dseg/slurm_out/3dseg-torch_predict-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_autoshear
##################################################################
##################################################################

ml release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1

## Display GPUs
nvidia-smi

source .venv_torch/bin/activate

echo $1
python ./scripts/train.py --config $1


exit 0