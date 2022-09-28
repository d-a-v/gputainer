#!/bin/bash
#SBATCH --job-name=test_cosypose
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

##SBATCH --gres=gpu:rtx_a6000:1
##SBATCH --gres=gpu:a30:1

if [ -z "${SLURM_JOB_NAME}" ]; then
    echo "This script must be run with sbatch:"
    echo "  sbatch $0"
    exit 1
fi

./20-runvnc
