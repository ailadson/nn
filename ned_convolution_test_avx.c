#include <assert.h>
#include <immintrin.h>
#include "matrix.h"
#include <stdio.h>
#include <time.h>

void convolve1d(float* input,
                float* kernel,
                float* destination,
                shape_t image_shape,
                shape_t kernel_shape) {

  assert(kernel_shape.height == 1);
  assert(kernel_shape.width == 3);

  assert(image_shape.width % 8 == 0);

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

  long t = clock();
  for (size_t i = 0; i < ITERS; i++) {
    convolve1d(input, kernel, destination, image_shape, kernel_shape);
  }
  t = clock() - t;

  printf("Result matrix!\n");
  print_matrix_corners(destination, image_shape);

  printf("time: %f\n", (double) t / CLOCKS_PER_SEC);

  return 0;
}
