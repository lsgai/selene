#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --constraint=v100
#SBATCH -n 1
#SBATCH -o train_mats_%j.out
#SBATCH -e train_mats_%j.err

source activate selene-env

python -u ../../../selene_cli.py ./train_deepsea_mat.yml --lr=0.08


CUDA_VISIBLE_DEVICES=6,7 python3 -u ../../../selene_cli.py train_deepsea_mat_danq_0.001.yml --lr 0.001 > train_deepsea_mat_danq_0.001.log

cd /local/datdb/selene/manuscript/case2/1_train_with_deepsea_mats
CUDA_VISIBLE_DEVICES=1 python3 -u ../../../selene_cli.py train_deepsea_mat_deepsea2015.yml --lr 0.08 > train_deepsea2017debug.log 

cd /local/datdb/selene/manuscript/case2/1_train_with_deepsea_mats
CUDA_VISIBLE_DEVICES=1,2 python3 -u ../../../selene_cli.py train_deepsea_mat_BertBase.yml --lr 0.0001 > train_BertBase.log 



scp -r selene/ lgai@hoffman2.idre.ucla.edu:/u/scratch/l/lgai

scp -r lgai@hoffman2.idre.ucla.edu:/u/scratch/l/lgai/selene .
