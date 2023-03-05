#!/bin/bash

#SBATCH --partition=day
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:00:00

module load miniconda
conda activate hypergen
python -u scripts/preprocess.py