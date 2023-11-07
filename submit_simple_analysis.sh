#!/bin/bash
#SBATCH --job-name=athenapk_analysis
#SBATCH --partition=mapu

# Load modules
module load openmpi/4.1.5 hdf5/1.14.1-2_openmpi-4.1.5_parallel Python/3.11.4

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
RUNDIR=Turb_nGPUs1_ncells256_accelrms1.0_B0.05_Adiab
OUTDIR=outputs/${RUNDIR}

# Run the sim
cd $PRJDIR
srun -N 1 -n 1 python3 scripts/run_analysis.py ${OUTDIR}
