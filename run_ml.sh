#!/bin/bash
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:1 --partition=batch

# salloc --nodes=1 --time=4:00:00 --gres=gpu:2 --res=hackathon2020
# salloc --nodes=1 --time=20:00:00 --gres=gpu:v100:2 --res=hackathon2020

################# for machine learning
# module load anaconda3/4.4.0
# module load tensorflow/2.0.0-cuda10.0-cudnn7.6-py3.7
# module load horovod/2019
# module load gpustack-default
# module load gpustack-legacy
# source ~/.bashrc_ibex
# conda activate t_env
# unset LD_LIBRARY_PATH
# srun python main.py
# source ~/.bashrc_ibex
# conda activate t_env4
module load dl
module load intelpython3
module load tensorflow/2.2
module load horovod/0.20.3
echo $LD_LIBRARY_PATH
module list

# srun ~/anaconda3/envs/t_env4/bin/python main.py
srun python main.py

# ls ./gpu_fwi/results/training_data_10_it_old_16_11/ | wc -l
# ls | wc -l
# ls ./datasets/dataset_vl_gen | wc -l
# git add F_modules.py F_models.py F_plotting.py F_utils.py keras_models main.py run_ml.sh ./datasets/dataset 