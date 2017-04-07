#!/usr/bin/env bash

set -e

# First build C modules
cd c
./build.sh
cd ..

# Then build Cython modules
cd pyx
./build.sh
cd ..

# Then link C modules and avx_convolve2d_py together.

unamestr=`uname`
if [[ "$unamestr" == "Darwin" ]]; then
    GCC=gcc-6
    #ANACONDA_DIR=/Users/anthonyladson/anaconda
    ANACONDA_DIR=/Users/ruggeri/.anaconda3
    BUILD_DIR=pyx/build/temp.macosx-10.7-x86_64-3.6

    $GCC -bundle -undefined dynamic_lookup \
         -L$ANACONDA_DIR/lib \
         -arch x86_64 \
         ./c/matrix.o ./c/avx_convolve2d.o \
         $BUILD_DIR/convolve2d.o \
         -o ./pyx/convolve2d.cpython-36m-darwin.so
elif [[ "$unamestr" == "Linux" ]]; then
    GCC=gcc
    ANACONDA_DIR=/home/ubuntu/.anaconda3
    BUILD_DIR=pyx/build/temp.linux-x86_64-3.6

    $GCC -pthread -shared -L$ANACONDA_DIR/lib \
         -Wl,-rpath=$ANACONDA_DIR/lib,--no-as-needed \
         c/matrix.o c/avx_convolve2d.o \
         $BUILD_DIR/convolve2d.o \
         -lpython3.6m \
         -o ./pyx/convolve2d.cpython-36m-x86_64-linux-gnu.so
else
    exit 1
fi
