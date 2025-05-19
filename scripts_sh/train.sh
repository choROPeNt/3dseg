#!/bin/bash

#Submit this script with: sbatch thefilename for the config yml

##################################################################
## SLURM Defintions
##################################################################
# walltime
#SBATCH --time=15:00:00                                  
# number of nodes
#SBATCH --nodes=1                                       
# number of tasks
#SBATCH --ntasks=1 
# number of processor cores (i.e. threads)                                     
#SBATCH --cpus-per-task=12     
 # specification of HPC partition
#SBATCH --partition=alpha      
 # memory per CPU core; max 16G per CPU
#SBATCH --mem-per-cpu=12G                       
# number of GPUs max 6 CPU per GPU on alpha
#SBATCH --gres=gpu:2     
# job name                               
#SBATCH -J "3dseg-BIIAX-train-%j"  
# output filepath for *.out file                     
#SBATCH --output=slurm_out/3dseg-BIIAX-train-%j.out  
# email address         
#SBATCH --mail-user=christian.duereth@tu-dresden.de   
# e-mail notifications  
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
# project for ressources
#SBATCH -A p_biiax                                      
##################################################################
##################################################################

# === Load Environment ===
ml release/24.04  GCC/12.3.0  OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1
source .venv_torch/bin/activate

# === Debug Info ===
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $SLURM_CPUS_PER_TASK"
nvidia-smi

# === Bind GPU explicitly ===
# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export PYTHONUNBUFFERED=1
# export NCCL_DEBUG=INFO
# export TORCH_USE_RTLD_GLOBAL=YES

# === Run Training ===
echo "Running config: $1"
python ./scripts/train.py --config "$1"

exit 0