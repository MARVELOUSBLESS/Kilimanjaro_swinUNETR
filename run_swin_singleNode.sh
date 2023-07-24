#!/bin/bash
#SBATCH --account=def-training-wa
# SBATCH --reservation hackathon-wr-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:8
#SBATCH --constraint=cascade,v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
module load python/3.9
source /home/guest183/run_swinUNETR_kilimanjaro/SWIN_ENV/bin/activate
# to solve the monai resolve_mode error 
# pip install monai[ignite]

# Brats 2021
# path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
# path_data='/scratch/guest183/BraTS_2021_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/'
# code to run the main method of the SWIN UNER network
# echo $path_swin'main.py'
# python $path_swin'main.py' --feature_size=48 --batch_size=1 --logdir=$path_swin'unetr_test_dir' --fold=0 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --save_checkpoint --val_every=10 --json_list=$path_swin'jsons/brats21_folds.json' --data_dir=$path_data --use_checkpoint --noamp
# python $path_swin'main.py' --json_list=$path_swin'jsons/brats21_folds.json' --data_dir=$path_data --val_every=5 --noamp --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --max_epochs=50 --logdir=$path_swin'unetr_test_dir'

# Brats 2023 Africa
path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
path_data='/scratch/guest183/BraTS_Africa_data/'
# code to run the main method of the SWIN UNER network
# echo $path_swin'main.py'
# to train on multiple GPU
python $path_swin'main.py' --fold=0 --smartcache_dataset --roi_x=128 --roi_y=128 --roi_z=128 --distributed --lrschedule='cosine_anneal' --json_list=$path_swin'jsons/brats23_africa_folds.json' --sw_batch_size=4 --batch_size=1 --data_dir=$path_data --val_every=20 --infer_overlap=0.7 --in_channels=4 --spatial_dims=3 --save_checkpoint --use_checkpoint --feature_size=48 --max_epochs=200 --logdir='rio_128_epoch_200_gpu_v100_32GB'

# --resume_ckpt --pretrained_dir=$path_swin'runs/4_gpu_60_epochs/' --pretrained_model_name='model_final.pt'
# 

# --roi_x=128 --roi_y=128 --roi_z=128

# --cache_dataset --save_checkpoint

# code to test the json generator
# python kfold_json_generator.py
