#!/bin/bash
#SBATCH -C gpu
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=0:10:00
#SBATCH --image=romerojosh/containers:sc21_tutorial
#SBATCH -J pm-crop64
#SBATCH -o %x-%j.out

args="${@}"

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500 # default from torch launcher

srun python analysis/datascaling_ddp ${args}