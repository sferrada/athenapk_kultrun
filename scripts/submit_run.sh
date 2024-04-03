#!/bin/bash
#SBATCH --job-name=turb_athenapk_gpu
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1

# Load modules
module load openmpi/4.1.5
module load gcc/12.2.0
module load hdf5/1.14.1-2_openmpi-4.1.5_parallel
module load cuda/12.2

# Set directory names
HOME_DIR=/home/sferrada
ATHENA_DIR=${HOME_DIR}/athenapk
KPATCH_DIR=${HOME_DIR}/athenapk_kultrun
SIM_DIR=NG_1-NC_256-TCOR_1.00-SOLW_1.00-ARMS_1.00-BINI_0.07-EOSG_1.00
OUT_DIR=${KPATCH_DIR}/outputs/${SIM_DIR}

# Run the sim
cd $ATHENA_DIR
mpirun ./build-host/bin/athenaPK -i ${OUT_DIR}/turbulence_philipp.in -d ${OUT_DIR}/ > "${OUT_DIR}/turbulence_philipp.out"

