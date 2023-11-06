#!/bin/bash
#SBATCH --job-name=athenapk_analysis
#SBATCH --partition=kurruf_gpu

# Load modules
module load openmpi/4.1.5 hdf5/1.14.1-2_openmpi-4.1.5_parallel Python/3.11.4

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
RUNDIR=Turb_nGPUs1_ncells256_accelrms1.0_B0.05_Adiab
OUTDIR=outputs/${RUNDIR}

# Run the sim
cd $PRJDIR
for X in `seq -w 00001 00049`; do srun -n 2 python3 ~/energy-transfer-analysis/run_analysis.py --res 256 --data_path ${OUTDIR}/$X.phdf --data_type AthenaPK --type flow --eos adiabatic --gamma 1.0001 --outfile ${OUTDIR}/flow-$X.hdf5 -forced; done
