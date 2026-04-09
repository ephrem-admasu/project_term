#!/bin/bash
#SBATCH --job-name=garbage-images
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=bigmem
#SBATCH --mem=128G
#SBATCH --time=0-23:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


module purge 

module load python/3.10  

source ~/my_evn/bin/activate
# python train_curb_demand.py
python train_curb_demand.py