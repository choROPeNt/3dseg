#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
#SBATCH --time=2:00:00                                 # walltime
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
python ./scripts/predict.py --config $1

=======
=======
>>>>>>> 629e93b (Re-initialize repository without old history.)
>>>>>>> 0d06ee2 (Re-initialize repository without old history.)
#!/bin/bash

#Submit this script with: sbatch thefilename

##################################################################
## SLURM Defintions
##################################################################
<<<<<<< HEAD
#SBATCH --time=3:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-task=12                               # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --mem-per-cpu=12G                             # memory per CPU core
#SBATCH --gres=gpu:2                                    # number of gpus
=======
#SBATCH --time=5:00:00                                 # walltime
#SBATCH --nodes=1                                       # number of nodes
#SBATCH --ntasks=1                                      # limit to one node
#SBATCH --cpus-per-gpu=6                               # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --mem=480G                             # memory per CPU core
#SBATCH --gres=gpu:4                                    # number of gpus
>>>>>>> c01128b (minor updates)
#SBATCH -J "3dseg-torch_predict"                           # job name
#SBATCH --output=slurm_out/3dseg-torch_pred-%j.out
#SBATCH --mail-user=christian.duereth@tu-dresden.de     # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_biiax
##################################################################
##################################################################

module load release/23.04 GCC/12.2.0 Python/3.10.8 OpenMPI/4.1.4 CUDA/11.8.0

## Display GPUs
nvidia-smi

source .venv_3dseg/bin/activate
echo $1
python ./scripts/predict.py --config $1

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> bf47220be6f13ee0507fdac5f20cdda293e61b5c
=======
>>>>>>> 629e93b (Re-initialize repository without old history.)
>>>>>>> 0d06ee2 (Re-initialize repository without old history.)
exit 0