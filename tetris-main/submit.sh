#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --job-name=Tetris_Video
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate tetris_ai

pip install imageio-ffmpeg opencv-python-headless

mkdir -p logs models

python -u train.py
