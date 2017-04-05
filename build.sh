#!/usr/bin/env bash

GCC=gcc-6
W_FLAGS="-Wall -Wextra -Werror"

$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
      ned_convolution_test_avx.c matrix.c -o ned_convolution_test_avx
$GCC -std=c11 -mavx -mavx2 -O3 $W_FLAGS \
      convolution.c matrix.c -o convolution
