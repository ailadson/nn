#include <stdbool.h>
#include <stddef.h>

#ifndef _MATRIX_H
#define _MATRIX_H

typedef struct {
  size_t height;
  size_t width;
} shape_t;

#define HEIGHT 28
#define WIDTH 28
#define KSIZE 3
#define ITERS 100000

float* allocate_matrix(shape_t shape);
float* build_example_input(shape_t image_shape);
float* build_example_kernel(shape_t kernel_shape);

void print_matrix(float* mat, shape_t shape);
void print_matrix_corners(float* mat, shape_t shape);

static inline float mat_get(float* matrix,
                            shape_t shape,
                            size_t i,
                            size_t j) {
  return matrix[i * shape.width + j];
}

static inline void mat_set(float* matrix,
                           shape_t shape,
                           size_t i,
                           size_t j,
                           float val) {
  matrix[i * shape.width + j] = val;
}

static inline float* mat_offset(float* matrix,
                                shape_t shape,
                                size_t i,
                                size_t j) {
  return matrix + (i * shape.width) + j;
}

// TODO: I'm not happy with this. Basically, if indexes are always
// size_t then they can never be negative. But then I'm a little
// concerned about when people subtract from indexes. Not a practical
// problem because (0 - 1 == 2**64-1), which is way bigger than any
// matrix dimension.
static inline bool row_in_bounds(shape_t shape, size_t row_idx) {
  return row_idx < shape.height;
}

static inline bool col_in_bounds(shape_t shape, size_t col_idx) {
  return col_idx < shape.width;
}

#endif // _MATRIX_H
