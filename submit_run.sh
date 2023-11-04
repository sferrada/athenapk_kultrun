#!/bin/bash
#SBATCH --job-name=turb_athenapk
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:A100:1

# Load modules, etc.
module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2

# Parse config file
source scripts/parse_yaml_file.sh
eval $(parse_yaml config.yaml)
EOS="${equation_of_state:0:5}"
EOS="${EOS^}"

# Set directory names
PRJDIR=/home/sferrada/athenapk_kultrun
RUNDIR=Turb_nGPUs1_ncells${number_of_cells}_accelrms${acceleration_field_rms}_B${initial_magnetic_field}_${EOS}

# Run the sim
cd $PRJDIR
mpirun ./athenapk/build-host/bin/athenaPK -i ./outputs/${RUNDIR}/turbulence_philipp.in -d ./outputs/${RUNDIR}/ > "./outputs/${RUNDIR}/turbulence_philipp.out"

# Run post-analysis if specified in the config file for a list of fields,
# e.g., fields_for_analysis=("mach_number" "density")
if [[ $run_analysis = "True" ]]; then
    fields_for_analysis=("mach_number")
    if [[ ${#fields_for_analysis[@]} -eq 0 ]]; then
        echo "No fields specified for analysis."
    else
        for field in "${fields_for_analysis[@]}"; do
            echo "Running analysis for field: $field"
            python3 scripts/get_average_value.py outputs/${RUNDIR} "$field"
        done
    fi
else
    echo "Post-run analysis is not enabled in the config."
fi



