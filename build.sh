#!/usr/bin/env bash

GCC=gcc-6

$GCC -std=c11 -mavx -mavx2 -O3 \
      ned_convolution_test_avx.c matrix.c -o ned_convolution_test_avx
$GCC -std=c11 -mavx -mavx2 -O3 \
      convolution.c matrix.c -o convolution
