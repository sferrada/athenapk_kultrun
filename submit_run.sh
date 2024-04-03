#!/bin/bash
#SBATCH --job-name=turb_athenapk_gpu
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1

# Load modules
module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2 Python/3.11.4

# Set directory names
HOME_DIR=/home/simonfch
REPO_DIR=/home/simonfch/athenapk_kultrun
SIM_DIR=NG_1-NC_064-TCOR_1.00-SOLW_1.00-ARMS_1.00-BINI_0.07-EOSG_1.00
OUT_DIR=outputs/${SIM_DIR}

# Run the sim
cd $REPO_DIR
mpirun ./athenapk/build-host/bin/athenaPK -i ./${OUT_DIR}/turbulence_philipp.in -d ./${OUT_DIR}/ > './${OUT_DIR}/turbulence_philipp.out'

