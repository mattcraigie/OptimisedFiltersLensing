#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=0:10:00
#SBATCH -J datascaling
#SBATCH -o logs/%x-%j.out

args="${@}"

# setup the python
source /global/common/software/nersc/pm-2022q3/sw/python/3.9-anaconda-2021.11/etc/profile.d/conda.sh
module load python
conda activate nbody

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)
export MASTER_PORT=49552

srun python analysis/datascaling_ddp.py ${args}