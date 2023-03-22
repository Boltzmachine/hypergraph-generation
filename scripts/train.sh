#!/bin/bash

#SBATCH --job-name=hypergraph
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/slurm
#SBATCH --mem=48G
#SBATCH --time 12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=v100

tango run config/shapenet.jsonnet -i src