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
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         gsub(/#[^"]*/, "", $3)  # Remove comments
         # Trim any leading or trailing spaces from the variable value
         value = gensub(/^[[:space:]]+|[[:space:]]+$/, "", "g", $3)
         printf("%s%s%s=\"%s\"\n", "'$prefix'", vn, $2, value);
      }
   }'
}

eval $(parse_yaml config.yaml)
RUNDIR=Turb_nGPUs1_ncells${number_of_cells}_accelrms${acceleration_field_rms}_B${initial_magnetic_field}_Adiab

# Run the sim
cd $PROJECTDIR
mpirun ./athenapk/build-host/bin/athenaPK -i ./outputs/${RUNDIR}/turbulence_philipp.in > "./outputs/${RUNDIR}/turbulence_philipp.out"

# After its done, move all output files to the correspondant folder
mv parthenon.* ./outputs/${RUNDIR}/

# mpirun ./athenapk/build-host/bin/athenaPK -i ./outputs/turb_nGPU1_nc64_M##_B0.05/turbulence_philipp_nc64.in > ./outputs/turbulence_philipp_nc64.out
# for M in 16 32 64 128; do
#   export N=$M;
#   ./bin/athenaPK -i ../inputs/linear_wave3d.in parthenon/meshblock/nx1=$((2*N)) parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N parthenon/mesh/nx1=$((2*M)) parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M
# done
