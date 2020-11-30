#!/bin/bash
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:1 --mem=128G --partition=batch
# salloc --nodes=1 --time=4:00:00 --gres=gpu:1 --partition=batch

# salloc --nodes=1 --time=4:00:00 --gres=gpu:2 --res=hackathon2020
# salloc --nodes=1 --time=4:00:00 --gres=gpu:v100:2 --res=hackathon2020

module list
echo $LD_LIBRARY_PATH
################# for machine learning
module load anaconda3/4.4.0
module load tensorflow/2.0.0-cuda10.0-cudnn7.6-py3.7
srun python main.py

# ls ./gpu_fwi/results/training_data_10_it_old_16_11/ | wc -l
# ls | wc -l
# ls ./datasets/dataset_vl_gen | wc -l
# git add F_modules.py F_models.py F_plotting.py F_utils.py keras_models main.py run_ml.sh ./datasets/dataset 