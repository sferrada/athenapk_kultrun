#!/bin/bash
#SBATCH --job-name=turb_athenapk_gpu
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1

# Load modules
module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
RUNDIR=NG_1-NC_256-TCOR_1.00-SOLW_1.00-ARMS_1.00-BINI_0.07-EOSG_1.00
OUTDIR=outputs/${RUNDIR}

# Run the sim
cd $PRJDIR
mpirun ./athenapk/build-host/bin/athenaPK -i ./${OUTDIR}/turbulence_philipp.in -d ./${OUTDIR}/ > "./${OUTDIR}/turbulence_philipp.out"
