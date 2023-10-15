#!/bin/bash
#SBATCH --job-name=test_athena_gpu
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G

module load python/3.10.11 cmake/3.21.1 gcc/12-20210426 cuda/12.1 openmpi/4.1.4_gcc-12-20210426_cuda-12.1 hdf5/1.12.2_openmpi-4.1.4_gcc-12-20210426_cuda-12.1_parallel
module unload gcc/9.5.0

# export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=32

cd /home/sferrada/athena_project/
mpirun ./athenapk/build-host/bin/athenaPK -i ./turbulence_philipp.in > ./turbulence_philipp.out
