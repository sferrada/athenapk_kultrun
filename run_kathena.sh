#!/bin/bash

KATHENADIR=/mnt/home/gretephi/src/kathena
N=64 # numerical resolution


# first test for Ms approx 0.5 solenoidal
ACCRMS=0.50 # with k=2 -> T = 1
TCORR=1.00
SOLW=1.00
TLIM=10.0 # should be 10 turnover times
DTHST=0.05
DTHDF=0.1

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.hydro -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..


# first test for Ms approx 1.0 solenoidal
ACCRMS=1.00 # with k=2 -> T = 0.5
TCORR=0.50
SOLW=1.00
TLIM=5.0 # should be 10 turnover times
DTHST=0.025
DTHDF=0.05

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.hydro -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..

# first test for Ms approx 1.0 solenoidal
ACCRMS=2.00 # with k=2 -> T = 0.25
TCORR=0.25
SOLW=1.00
TLIM=2.5 # should be 10 turnover times
DTHST=0.0125
DTHDF=0.025

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.hydro -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..



# first test for Ms approx 2 mixed
ACCRMS=2.00 # with k=2 -> T = 0.25
TCORR=0.25
SOLW=0.50
TLIM=2.50 # should be 10 turnover times
DTHST=0.0125
DTHDF=0.025

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.hydro -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..

# first test for Ms approx 2 compressive
ACCRMS=2.00 # with k=2 -> T = 0.25
TCORR=0.25
SOLW=0.00
TLIM=2.50 # should be 10 turnover times
DTHST=0.0125
DTHDF=0.025

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.hydro -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..



# first test for some parameters in between 
ACCRMS=1.25 # with k=2 -> T = 0.4
TCORR=0.2 # -> 0.5T
SOLW=0.30
TLIM=4.0 # should be 10 turnover times
DTHST=0.02
DTHDF=0.04

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.hydro -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..

# same run with more  proc 
ACCRMS=1.25 # with k=2 -> T = 0.4
TCORR=0.2 # -> 0.5T
SOLW=0.30
TLIM=4.0 # should be 10 turnover times
DTHST=0.02
DTHDF=0.04

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW-NPROC_8
mkdir $DIR
cd $DIR
srun -N 1 -n 8 $KATHENADIR/bin/athena.cuda.hydro -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$((N/2))  meshblock/nx2=$((N/2)) meshblock/nx3=$((N/2)) problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..

# parameters in between restarted
ACCRMS=1.25 # with k=2 -> T = 0.4
TCORR=0.2 # -> 0.5T
SOLW=0.30
TLIM=4.0 # should be 10 turnover times
DTHST=0.02
DTHDF=0.04

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW-RST
mkdir $DIR
cd $DIR
cp ../T_$TCORR-A_$ACCRMS-S_$SOLW/Turb.00001.rst .
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.hydro -r Turb.00001.rst  |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..

# parameters in between MHD
ACCRMS=1.25 # with k=2 -> T = 0.4
TCORR=0.2 # -> 0.5T
SOLW=0.30
TLIM=4.0 # should be 10 turnover times
DTHST=0.02
DTHDF=0.04

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW-MHD
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.cuda.mhd -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..


# parameters in between MHD
ACCRMS=1.25 # with k=2 -> T = 0.4
TCORR=0.2 # -> 0.5T
SOLW=0.30
TLIM=4.0 # should be 10 turnover times
DTHST=0.02
DTHDF=0.04

DIR=T_$TCORR-A_$ACCRMS-S_$SOLW-GCC
mkdir $DIR
cd $DIR
srun -N 1 -n 1 $KATHENADIR/bin/athena.gcc.mhd -i $KATHENADIR/inputs/mhd/athinput.fmturb output1/dt=$DTHST output2/dt=$DTHDF output4/dt=$DTHDF output3/dt=$DTHDF time/tlim=$TLIM mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$N  meshblock/nx2=$N meshblock/nx3=$N problem/accel_rms=$ACCRMS problem/corr_time=$TCORR problem/sol_weight=$SOLW |tee ath.out
for X in `seq -w 0001 0100`; do srun -n 2 python ~/src/energy-transfer-analysis/run_analysis.py --res $N --data_path 0$X.athdf  --data_type AthenaPP --type flow --eos adiabatic --gamma 1.0001 --outfile flow-$X.hdf5 -forced; done
cd ..
