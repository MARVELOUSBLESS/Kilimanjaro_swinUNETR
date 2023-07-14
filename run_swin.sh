#!/bin/bash
#SBATCH --account=def-training-wa
# SBATCH --reservation hackathon-wr-gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=t4:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
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
python $path_swin'main.py' --world_size=2 --distributed --lrschedule='cosine_anneal' --json_list=$path_swin'jsons/brats23_africa_folds.json' --sw_batch_size=4 --batch_size=2 --data_dir=$path_data --val_every=60 --infer_overlap=0.7 --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48 --max_epochs=60 --logdir='4_gpu_60_epochs_cached'

# --smartcache_dataset --save_checkpoint
# this is a code to do Quality assuarance on a trained model 
# python quality_assurance.py
# code to test the json generator
# python kfold_json_generator.py

# code to test the dataloader.py
# python dataloader_baraka.py

# python -c "import monai" || pip install -q "monai-weekly[nibabel]"
# python -c "import matplotlib" || pip install -q matplotlib


# source /home/guest183/hackathon/bin/activate
# making sure nnunet python module is installed
# echo $(python -c "import nnunet; print(nnunet.__version_))")
# run training
# nnUNet_train "3d_fullres" "nnUNetTrainerV2BraTSRegions_DA4_BN_BD" 500 5 --npz # BL config
# nnUNet_train 3d_fullres nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm <TASK_ID> <FOLD> --npz # BL + L + GN config

# exit