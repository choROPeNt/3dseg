#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=16:00:00                 # walltime
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # limit to one node
#SBATCH --cpus-per-task=6               # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --mem-per-cpu=40000                 # memory per CPU core
#SBATCH --gres=gpu:1                    # number of gpus
#SBATCH -J "pytorch_Keypoint_DCB"       # job name
#SBATCH --mail-user=christian.duereth@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_autoshear
##################################################################
##################################################################

module load modenv/hiera GCCcore/10.2.0 Python/3.8.6 CUDA/11.3.1

source /home/h2/dchristi/alpha_Py386_CU116/bin/activate

python train_model.py

exit 0