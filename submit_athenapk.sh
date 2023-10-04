#!/bin/bash
#SBATCH --job-name=test_athena_gpu
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:A100:1

module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2

cd /home/sferrada/athenapk_kultrun/
mpirun ./athenapk/build-host/bin/athenaPK -i ./inputs/turbulence_philipp_64.in > ./outputs/turbulence_philipp.out
