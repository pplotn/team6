#!/bin/bash 
#SBATCH --job-name=multi
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=v100
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G


module load dl
module load intelpython3
# or tensorflow
module load tensorflow/2.2
module load horovod/0.20.3
module list

export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=tcp
#   horovod benchmark
# srun -u -n ${SLURM_NTASKS} -N ${SLURM_NNODES} -c ${SLURM_CPUS_PER_TASK} --cpu-bind=cores  python tensorflow2_keras_mnist.py 
#   horovod benchmark
srun -u -n ${SLURM_NTASKS} -N ${SLURM_NNODES} -c ${SLURM_CPUS_PER_TASK} --cpu-bind=cores  python main_multi.py 