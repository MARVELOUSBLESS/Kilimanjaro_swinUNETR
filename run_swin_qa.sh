#!/bin/bash
#SBATCH --account=def-training-wa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module load python/3.9
source /home/guest183/run_swinUNETR_kilimanjaro/SWIN_ENV/bin/activate

path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
path_data='/scratch/guest183/BraTS_Africa_data/'

# # generate model predictions
python $path_swin'test.py'  --infer_overlap=0.7\
 --data_dir=$path_data --exp_name='epoch100_baseModel_GLI_test'\
 --json_list=$path_swin'jsons/brats23_gli_remainig_test.json'\
 --pretrained_dir=$path_swin'pretrained_models/'\
 --pretrained_model_name='model-epoch100-baseModel-2023.pt'

# # run quality assurance on model predictions
# python $path_swin'quality_assurance.py'