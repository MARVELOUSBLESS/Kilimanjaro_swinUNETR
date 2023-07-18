#!/bin/bash
#SBATCH --account=def-training-wa
#SBATCH --nodes=2
#SBATCH --gpus-per-node=t4:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00

module load python/3.9
source /home/guest183/run_swinUNETR_kilimanjaro/SWIN_ENV/bin/activate

