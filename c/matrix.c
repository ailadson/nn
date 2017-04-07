#include "matrix.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float* allocate_matrix(shape_t shape) {
  size_t bytes = shape.height * shape.width * sizeof(float);
  float* mat = malloc(bytes);
  memset(mat, 0, bytes);
  return mat;
}

float* build_example_input(shape_t image_shape) {
  float* input = allocate_matrix(image_shape);
  for (size_t i = 0; i < image_shape.height; i++) {
    for (size_t j = 0; j < image_shape.width; j++) {
      mat_set(input, image_shape, i, j, (i + j));
    }
  }

  return input;
}

float* build_example_kernel(shape_t kernel_shape) {
  float* kernel = allocate_matrix(kernel_shape);

  assert(kernel_shape.height == 3);
  assert(kernel_shape.width == 3);

  for (size_t i = 0; i < kernel_shape.height; i++) {
    for (size_t j = 0; j < kernel_shape.width; j++) {
      float val = 0.0;
      if ((i == 0) || (i == kernel_shape.height - 1)) {
        val = 1.0;
      } else if ((j == 0) || (j == kernel_shape.width - 1)) {
        val = 1.0;
      }
      mat_set(kernel, kernel_shape, i, j, val);
    }
  }

  return kernel;
}

void print_matrix(float* mat, shape_t shape) {
  for (size_t i = 0; i < shape.height; i++) {
    for (size_t j = 0; j < shape.width; j++) {
      printf("%6.2f ", mat_get(mat, shape, i, j));
    }

    printf("\n");
  }
}

void print_matrix_corners(float* mat, shape_t shape) {
  printf("%6.2f ", mat_get(mat, shape, 0, 0));
  printf("%6.2f ", mat_get(mat, shape, 0, shape.width - 1));
  printf("%6.2f ", mat_get(mat, shape, shape.height - 1, 0));
  printf("%6.2f ", mat_get(mat, shape, shape.height - 1, shape.width - 1));
  printf("\n");
}
