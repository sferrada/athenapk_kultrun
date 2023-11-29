#!/bin/bash
#SBATCH --job-name=athenapk_analysis
#SBATCH --partition=mapu

# Load modules
module load openmpi/4.1.5 fftw/3.3.10_openmpi-4.1.5 hdf5/1.14.1-2_openmpi-4.1.5_parallel Python/3.11.4

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
OUTDIR=outputs
cd $PRJDIR

# Iterate over folders matching the pattern "NG_1-NC*"
for folder in $(find . -type d -name "NG_1-NC*"); do
    # Run simple post-analysis for each folder
    python3 scripts/run_analysis.py "${folder}"
done

