#!/bin/bash

if ! [  -n "$(uname -a | grep Ubuntu)" ]; then
  module load compilers/cmake/3.21.4-gcc-11.2.0
  module load compilers/gcc-11.2.0
  module load mpi/openmpi-4.1.1
fi

cd ../alya-raise/build
rm CMakeCache.txt

cmake ..

make clean

CORES=$(getconf _NPROCESSORS_ONLN)
echo "Number of CPU/cores used for build: $CORES"

make -j$CORES
#make install

cp src/alya/alya ../../alya-workflow/bin
cp src/alya2pos/alya2pos ../../alya-workflow/bin
