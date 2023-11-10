#!/bin/bash
#SBATCH --job-name=athenapk_analysis
#SBATCH --partition=mapu

# Load modules
module load openmpi/4.1.5 fftw/3.3.10_openmpi-4.1.5 hdf5/1.14.1-2_openmpi-4.1.5_parallel Python/3.11.4

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
RUNDIR=Turb_nGPUs1_ncells256_accelrms1.0_B0.05_Adiab
OUTDIR=outputs/${RUNDIR}

# Run the sim
cd $PRJDIR
source /home/sferrada/miniconda3/bin/activate myenv
for X in `seq -w 00001 00049`; do mpirun -N 1 -n 1 python3 ~/energy-transfer-analysis/run_analysis.py --res 256 --data_path ${OUTDIR}/parthenon.prim.${X}.phdf --data_type AthenaPK --type flow --eos adiabatic --gamma 1.0001 --outfile ${OUTDIR}/flow.$X.hdf5 -forced; done
