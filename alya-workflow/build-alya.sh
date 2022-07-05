#!/bin/bash

if ! [  -n "$(uname -a | grep Ubuntu)" ]; then
  module load spack/cmake/3.21.4-gcc-11.2.0
  module load spack/gcc-11.2.0
  module load mpi/openmpi-4.1.1
fi

cd ../alya-raise/build

cmake ..

make clean
make -j8
make install

cp src/alya/alya ../../alya-workflow/bin
cp src/alya2pos/alya2pos ../../alya-workflow/bin
