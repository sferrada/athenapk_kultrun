#!/bin/bash
#SBATCH --job-name=athenapk_analysis
#SBATCH --partition=mapu

# Load modules
module load openmpi/4.1.5 fftw/3.3.10_openmpi-4.1.5 hdf5/1.14.1-2_openmpi-4.1.5_parallel Python/3.11.4

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
RUNDIR=NG_1-NC_256-TCOR_1.00-SOLW_1.00-ARMS_1.00-BINI_0.30-EOSG_1.00
OUTDIR=outputs/${RUNDIR}
cd $PRJDIR

# Run simple post-analysis
python3 scripts/run_analysis.py ${OUTDIR}
