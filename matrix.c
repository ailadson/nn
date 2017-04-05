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

float mat_get(float* matrix, shape_t shape, int i, int j) {
  return matrix[i * shape.width + j];
}

void mat_set(float* matrix, shape_t shape, int i, int j, float val) {
  matrix[i * shape.width + j] = val;
}

float* mat_offset(float* matrix, shape_t shape, int i, int j) {
  return matrix + (i * shape.width) + j;
}

float* build_example_input(shape_t image_shape) {
  float* input = allocate_matrix(image_shape);
  for (int i = 0; i < image_shape.height; i++) {
    for (int j = 0; j < image_shape.width; j++) {
      mat_set(input, image_shape, i, j, (i + j));
    }
  }

  return input;
}

float* build_example_kernel(shape_t kernel_shape) {
  float* kernel = allocate_matrix(kernel_shape);

  assert(kernel_shape.height == 1);
  assert(kernel_shape.width == 3);

  mat_set(kernel, kernel_shape, 0, 0, 1.0);
  mat_set(kernel, kernel_shape, 0, 1, 0.0);
  mat_set(kernel, kernel_shape, 0, 2, 1.0);

  return kernel;
}

void print_matrix(float* mat, shape_t shape) {
  for (int i = 0; i < shape.height; i++) {
    for (int j = 0; j < shape.width; j++) {
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

bool row_in_bounds(shape_t shape, int row_idx) {
  return ((row_idx >= 0) && (row_idx < shape.height));
}
