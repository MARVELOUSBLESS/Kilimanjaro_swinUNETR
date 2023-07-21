#!/bin/bash
#SBATCH --account=def-training-wa
# SBATCH --reservation hackathon-wr-gpu
#SBATCH --nodes=1
# SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:15:00

module load python/3.9
source /home/guest183/run_swinUNETR_kilimanjaro/SWIN_ENV/bin/activate

path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
path_data='/scratch/guest183/BraTS_Africa_data/'

# generate model predictions
# python $path_swin'test.py' --data_dir=$path_data --exp_name='4gpu_120_epoch' --json_list=$path_swin'jsons/brats23_africa_validation_folds.json' --pretrained_dir=$path_swin'runs/4_gpu_120_epochs' --pretrained_model_name='model_final.pt' --infer_overlap=0.7

# run quality assurance on model predictions
python $path_swin'quality_assurance.py'