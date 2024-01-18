#!/bin/bash
#SBATCH --job-name=athenapk_analysis
#SBATCH --partition=mapu

# Load modules
module load openmpi/4.1.5 fftw/3.3.10_openmpi-4.1.5 hdf5/1.14.1-2_openmpi-4.1.5_parallel Python/3.11.4

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
OUTDIR=outputs

# Loop through each run directory
for RUNDIR in "${PRJDIR}/${OUTDIR}"/*/; do
    # Check if it's a directory
    if [ -d "$RUNDIR" ]; then
        # Get the run name from the directory
        RUN_NAME=$(basename "$RUNDIR")

        # Check if analysis.h5 already exists in the run directory
        if [ -e "${PRJDIR}/${OUTDIR}/${RUN_NAME}/analysis.h5" ]; then
            echo "Analysis already completed for ${RUN_NAME}. Skipping."
        else
            # Run the analysis script with the full path
            python3 "${PRJDIR}/athenapk_kultrun.py" analyse --run "${PRJDIR}/${OUTDIR}/${RUN_NAME}"
        fi
    fi
done

