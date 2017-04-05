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

float mat_get(float* matrix, shape_t shape, size_t i, size_t j) {
  return matrix[i * shape.width + j];
}

void mat_set(float* matrix, shape_t shape, size_t i, size_t j, float val) {
  matrix[i * shape.width + j] = val;
}

float* mat_offset(float* matrix, shape_t shape, size_t i, size_t j) {
  return matrix + (i * shape.width) + j;
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

// TODO: I'm not happy with this. Basically, if indexes are always
// size_t then they can never be negative. But then I'm a little
// concerned about when people subtract from indexes. Not a practical
// problem because (0 - 1 == 2**64-1), which is way bigger than any
// matrix dimension.
bool row_in_bounds(shape_t shape, size_t row_idx) {
  return row_idx < shape.height;
}

bool col_in_bounds(shape_t shape, size_t col_idx) {
  return col_idx < shape.width;
}
