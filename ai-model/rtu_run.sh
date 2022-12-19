#!/bin/sh

echo HOSTNAME=$(hostname) TASKNUM=$PBS_TASKNUM NODENUM=$PBS_NODENUM

module load conda
source activate raise_windturbine

module load cuda/cuda-10.2

cd $PBS_O_WORKDIR

STARTTIME=$(date +%s)

echo "Starting windturbine AI task..."

../ai-model/model_surrogate/run_train_surrogate_local.sh

ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to run AI model"
