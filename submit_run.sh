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
PRJDIR=/home/sferrada/athenapk_kultrun
RUNDIR=Turb_nGPUs1_ncells128_accelrms0.1_B0.05_Adiab
OUTDIR=outputs/${RUNDIR}

# Run the sim
cd $PRJDIR
mpirun ./athenapk/build-host/bin/athenaPK -i ./${OUTDIR}/turbulence_philipp.in -d ./${OUTDIR}/ > "./${OUTDIR}/turbulence_philipp.out"

# Run post-analysis if specified in the config file for a list of fields
run_analysis=True
fields_for_analysis=("mach_number")

if [[ $run_analysis = "True" ]]; then
    if [[ ${#fields_for_analysis[@]} -eq 0 ]]; then
        echo "No fields specified for analysis."
    else
        for field in "${fields_for_analysis[@]}"; do
            echo "Running analysis for field: $field"
            python3 scripts/get_average_value.py ${OUTDIR} "$field"
        done
    fi
else
    echo "Post-run analysis is not enabled in the config."
fi
