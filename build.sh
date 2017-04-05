#!/usr/bin/env bash

gcc-6 -std=c11 -mavx -mavx2 ned_convolution_test_avx.c matrix.c -o ned_convolution_test_avx
gcc-6 -std=c11 -mavx -mavx2 convolution.c matrix.c -o convolution
