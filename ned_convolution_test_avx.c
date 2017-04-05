#include <assert.h>
#include "immintrin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int height;
  int width;
} shape_t;

#define HEIGHT 512
#define WIDTH 512
#define KSIZE 3

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

void convolve1d(float* input,
                float* kernel,
                float* destination,
                shape_t image_shape,
                shape_t kernel_shape) {

  assert(kernel_shape.height == 1);
  assert(kernel_shape.width == 3);

  float k0_arr[8] = { [0 ... 7] = mat_get(kernel, kernel_shape, 0, 0) };
  float k1_arr[8] = { [0 ... 7] = mat_get(kernel, kernel_shape, 0, 1) };
  float k2_arr[8] = { [0 ... 7] = mat_get(kernel, kernel_shape, 0, 2) };
  __m256 k0_avx = _mm256_loadu_ps(k0_arr);
  __m256 k1_avx = _mm256_loadu_ps(k1_arr);
  __m256 k2_avx = _mm256_loadu_ps(k2_arr);

  // Hack to get a float of ones.
  int neg_one = 0xffffffff;
  float ones_float = *((float*) (&neg_one));

  __m256i right_shift_avx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
  float drop_left_el_arr[] = { 0x0000, [1 ... 7] = ones_float };
  __m256 drop_left_el_avx = _mm256_loadu_ps(drop_left_el_arr);

  __m256i left_shift_avx = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
  float drop_right_el_arr[] = { [0 ... 6] = ones_float, 0x00 };
  __m256 drop_right_el_avx = _mm256_loadu_ps(drop_right_el_arr);

  __m256 data_avx;
  __m256 result_avx;
  __m256 prod0_avx;
  __m256 prod1_avx;
  __m256 prod2_avx;

  for (int i = 0; i < image_shape.height; i++) {
    for (int j = 0; j < image_shape.width; j += 8) {
      data_avx = _mm256_loadu_ps(mat_offset(input, image_shape, i, j));
      prod0_avx = _mm256_mul_ps(k0_avx, data_avx);
      prod1_avx = _mm256_mul_ps(k1_avx, data_avx);
      prod2_avx = _mm256_mul_ps(k2_avx, data_avx);

      prod0_avx = _mm256_permutevar8x32_ps(prod0_avx, right_shift_avx);
      prod0_avx = _mm256_and_ps(prod0_avx, drop_left_el_avx);
      prod2_avx = _mm256_permutevar8x32_ps(prod2_avx, left_shift_avx);
      prod2_avx = _mm256_and_ps(prod2_avx, drop_right_el_avx);

      result_avx = _mm256_add_ps(prod0_avx, prod1_avx);
      result_avx = _mm256_add_ps(result_avx, prod2_avx);

      float* destination_offset = mat_offset(destination, image_shape, i, j);
      _mm256_storeu_ps(destination_offset, result_avx);
    }
  }
}

int main() {
  shape_t image_shape = { .height = HEIGHT, .width = WIDTH };
  float* input = build_example_input(image_shape);
  float* destination = allocate_matrix(image_shape);
  // TODO: later implement for 2d kernel.
  shape_t kernel_shape = { .height = 1, .width = KSIZE };
  float* kernel = build_example_kernel(kernel_shape);

  printf("Input matrix!\n");
  print_matrix_corners(input, image_shape);
  printf("Kernel matrix!\n");
  print_matrix(kernel, kernel_shape);

  convolve1d(input, kernel, destination, image_shape, kernel_shape);
  printf("Result matrix!\n");
  print_matrix_corners(destination, image_shape);

  return 0;
}
