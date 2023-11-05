#!/bin/bash
#SBATCH --job-name=turb_athenapk_gpu
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:A100:1

# Load modules
module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2

# Set directory names
PRJDIR=/home/simonfch/athenapk_kultrun
RUNDIR=Turb_nGPUs1_ncells128_accelrms0.1_B0.05_Adiab
OUTDIR=outputs/${RUNDIR}

# Run the sim
cd $PRJDIR
mpirun ./athenapk/build-host/bin/athenaPK -i ./${OUTDIR}/turbulence_philipp.in -d ./${OUTDIR}/ > "./${OUTDIR}/turbulence_philipp.out"

# Run post-analysis if specified in the config file
python3 scripts/run_analysis.py ${OUTDIR}
