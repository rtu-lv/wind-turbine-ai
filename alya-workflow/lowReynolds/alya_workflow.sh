#!/bin/sh

echo HOSTNAME=$(hostname) TASKNUM=$PBS_TASKNUM NODENUM=$PBS_NODENUM

source activate raise_wind

module load spack/gcc-11.2.0
module load mpi/openmpi-4.1.1

cd $PBS_O_WORKDIR

STARTTIME=$(date +%s)

python3 parallel.py

ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to complete alya task..."
