#!/bin/bash
#SBATCH --job-name=test_cosypose
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt
#SBATCH --mem=50G

##SBATCH --gres=gpu:${gpu}

./20-runvnc
