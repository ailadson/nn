#!/usr/bin/env bash

GCC=gcc-6
W_FLAGS="-Wall -Wextra -Werror"

$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS -fPIC -c avx_convolve2d.c
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS -fPIC -c matrix.c
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
     avx_convolve2d_main.c avx_convolve2d.o matrix.o -o avx_convolve2d
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
      convolution.c matrix.c -o convolution

cd avx_convolve2d_py
python setup.py build_ext --inplace
cd ..

# OSX version
ANACONDA_DIR=/Users/anthonyladson/anaconda
LIB_PYTHON=$ANACONDA_DIR/lib-lpython3.6m
BUILD_DIR=avx_convolve2d_py/build/temp.macosx-10.7-x86_64-3.6

$GCC -bundle -undefined dynamic_lookup \
     -L/Users/anthonyladson/anaconda/lib \
     -L/Users/anthonyladson/anaconda/lib \
     -arch x86_64 \
     $BUILD_DIR/main.o \
     matrix.o avx_convolve2d.o \
     -L/Users/anthonyladson/anaconda/lib \
     -o avx_convolve2d_py/main.cpython-36m-darwin.so

# Linux version
#ANACONDA_DIR=/home/ubuntu/.anaconda3
#BUILD_DIR=avx_convolve2d_py/build/temp.linux-x86_64-3.6

#$GCC -pthread -shared -L$ANACONDA_DIR/lib \
#     -Wl,-rpath=$ANACONDA_DIR/lib,--no-as-needed $BUILD_DIR/main.o \
#     matrix.o avx_convolve2d.o \
#     -L$ANACONDA_DIR/lib-lpython3.6m \
#     -o avx_convolve2d_py/main.cpython-36m-x86_64-linux-gnu.so
