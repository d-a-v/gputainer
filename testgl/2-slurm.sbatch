#!/bin/bash
#SBATCH --job-name=testgl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --output=stdout-%j.txt
#SBATCH --error=stderr-%j.txt
#SBATCH --mem=5G
#SBATCH --gres=gpu:1

##SBATCH --gres=gpu:rtx_a6000:1
##SBATCH --gres=gpu:a30:1

if [ -z "${SLURM_JOB_NAME}" ]; then
    echo "This script must be run with sbatch:"
    echo "  sbatch $0"
    exit 1
fi

./2-runvnc "$@"
