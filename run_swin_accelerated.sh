#!/bin/bash
#SBATCH --account=def-training-wa
#SBATCH --nodes=2
#SBATCH --gpus-per-node=t4:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00

export HEAD_NODE=$(hostname) # store head node's address
export HEAD_NODE_PORT=34567 # choose a port on the main node to start accelerate's main process

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO  # allow for nccl debugging


echo "Node $SLURM_NODEID says: main node at $HEAD_NODE"
echo "Node $SLURM_NODEID says: Launching python script with accelerate..."


# Brats 2023 Africa
path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
path_data='/scratch/guest183/BraTS_Africa_data/'


accelerate launch \
--multi_gpu \
--gpu_ids="all" \
--num_machines=$SLURM_NNODES \
--machine_rank=$SLURM_NODEID \
--num_processes=8 \ # This is the total number of GPUs across all nodes
--main_process_ip="$HEAD_NODE" \
--main_process_port=$HEAD_NODE_PORT \
$path_swin'main_accelerated.py' --distributed --use_checkpoint --lrschedule='cosine_anneal' --json_list=$path_swin'jsons/brats23_africa_folds.json' --sw_batch_size=4 --batch_size=2 --data_dir=$path_data --val_every=40 --infer_overlap=0.7 --in_channels=4 --spatial_dims=3 --feature_size=48 --max_epochs=80 --logdir='8_gpu_80_epochs'

# $path_swin'main_accelerated.py' --world_size=1 --distributed --use_checkpoint --lrschedule='cosine_anneal' --json_list=$path_swin'jsons/brats23_africa_folds.json' --sw_batch_size=4 --batch_size=2 --data_dir=$path_data --val_every=60 --infer_overlap=0.7 --in_channels=4 --spatial_dims=3 --feature_size=48 --max_epochs=80 --logdir='8_gpu_80_epochs'