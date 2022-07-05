#!/bin/sh
#PBS -N alya_workflow
#PBS -q batch
#PBS -A coe_raise
#PBS -l nodes=1:ppn=4,feature=rudens
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -V

#NODES=($(cat $PBS_NODEFILE | sort | uniq))
#echo $NODES

pbsdsh -v "$PBS_O_WORKDIR/alya_workflow.sh"

#cd $PBS_O_WORKDIR
#./alya_workflow.sh
