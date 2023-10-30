#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

# SBATCH --time=00:15:00             # walltime limit (HH:MM:SS)
# SBATCH --nodes=1                  # number of nodes
# SBATCH --ntasks-per-node=8        # 8 processor core(s) per node 
# SBATCH --gres=gpu:a100:4          # specify GPU model and count
# SBATCH --partition=class-gpu-short  # specify partition
# SBATCH --account=class-faculty    # specify slurm account
# SBATCH --reservation=multGPU      # specify reservation name
# SBATCH --output=slurm-%j.out      # specify output filename

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniconda3
source activate dlrg

python ddp.py -g 4
