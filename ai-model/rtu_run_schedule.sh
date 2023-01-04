#!/bin/sh
#PBS -N windturbine_ai
#PBS -q batch
#PBS -A coe_raise
#PBS -l nodes=1:ppn=4,feature=a100
#PBS -l walltime=01:00:00
#PBS -j oe

$PBS_O_WORKDIR/rtu_run.sh
