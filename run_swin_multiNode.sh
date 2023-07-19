#!/bin/bash
#SBATCH --account=def-training-wa
#SBATCH --nodes=2
#SBATCH --gpus-per-node=t4:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

module load python/3.9
source /home/guest183/run_swinUNETR_kilimanjaro/SWIN_ENV/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export NCCL_DEBUG=INFO  # allow for nccl debugging
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

# Brats 2023 Africa
path_swin='/home/guest183/research-contributions/SwinUNETR/BRATS21/'
path_data='/scratch/guest183/BraTS_Africa_data/'


srun python $path_swin'main.py' --world_size=2 --distributed --use_checkpoint \ 
    --lrschedule='cosine_anneal' --dist-url="tcp://$MASTER_ADDR:3456" \ 
    --json_list=$path_swin'jsons/brats23_africa_folds.json' --sw_batch_size=4 --batch_size=2 \
    --data_dir=$path_data --val_every=60 --infer_overlap=0.7 --in_channels=4 --spatial_dims=3 \
    --feature_size=48 --max_epochs=60 --logdir='8_gpu_80_epochs'