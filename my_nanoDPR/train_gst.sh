#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate train_dpr
hostname
conda info
accelerate launch \
    --gpu_ids 0 \
    train_dpr.py \
    --config_file config/gst_24G_train_dpr_nq.yaml
conda deactivate 