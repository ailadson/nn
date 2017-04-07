#!/usr/bin/env bash

set -e

unamestr=`uname`
if [[ "$unamestr" == "Darwin" ]]; then
    GCC=gcc-6
elif [[ "$unamestr" == "Linux" ]]; then
    GCC=gcc
else
    exit 1
fi

W_FLAGS="-Wall -Wextra -Werror"

$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS -fPIC -c avx_convolve2d.c
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS -fPIC -c matrix.c
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
     avx_convolve2d_main.c avx_convolve2d.o matrix.o -o avx_convolve2d
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
      convolution.c matrix.c -o convolution
