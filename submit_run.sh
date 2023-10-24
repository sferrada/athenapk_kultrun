#!/bin/bash
#SBATCH --job-name=turb_athenapk
#SBATCH --partition=kurruf_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:A100:1

# Load modules
module load openmpi/4.1.5 gcc/12.2.0 hdf5/1.14.1-2_openmpi-4.1.5_parallel cuda/12.2

# Set variables
PROJECTDIR=/home/sferrada/athenapk_kultrun
ATHENAPKBIN=$PROJECTDIR/athenapk/build-host/bin/athenaPK

# Read config file hydro variables
eval $(parse_yaml config.yaml)
RUNDIR=Turb_nGPUs1_ncells${number_of_cells}_accelrms${acceleration_field_rms}_B${initial_magnetic_field}_Adiab

# Run the sim
cd $PROJECTDIR
mpirun $ATHENAPKBIN -i ./outputs/$RUNDIR/turbulence_philipp.in > ./outputs/$RUNDIR/turbulence_philipp.out

# mpirun ./athenapk/build-host/bin/athenaPK -i ./outputs/turb_nGPU1_nc64_M##_B0.05/turbulence_philipp_nc64.in > ./outputs/turbulence_philipp_nc64.out
# # to run a convergence test:
# for M in 16 32 64 128; do
#   export N=$M;
#   ./bin/athenaPK -i ../inputs/linear_wave3d.in parthenon/meshblock/nx1=$((2*N)) parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N parthenon/mesh/nx1=$((2*M)) parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M
# done