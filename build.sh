#!/usr/bin/env bash

GCC=gcc-6
W_FLAGS="-Wall -Wextra -Werror"

$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
     avx_convolve2d.c avx_convolve2d_main.c matrix.c -o avx_convolve2d
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
      convolution.c matrix.c -o convolution
