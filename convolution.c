#include <assert.h>
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

float* ipt;
float* kernel;
float* des;

void convolve2d(
  float* ipt,
  float* kernel,
  float* target,
  shape_t kernel_s,
  shape_t image_s);

float sum_of_pixels(float* image, shape_t image_s);

int main() {
  printf("Number of bytes in float: %ld\n", sizeof(float));
  shape_t kernel_s = { .width = KSIZE, .height = 1 };
  shape_t image_s = { .width = WIDTH, .height = HEIGHT };

  ipt = build_example_input(image_s);
  des = allocate_matrix(image_s);
  kernel = build_example_kernel(kernel_s);

  long t = clock();
  for (size_t i = 0; i < ITERS; i++) {
    convolve2d(ipt, kernel, des, kernel_s, image_s);
  }
  t = clock() - t;
  print_matrix_corners(des, image_s);

  printf("time: %f\n", (double) t / CLOCKS_PER_SEC);
}

void convolve_step(
  float* ipt,
  float* target,
  float kernel_val,
  int kernel_offset_i,
  int kernel_offset_j,
  shape_t image_s);

void convolve2d(
  float* ipt,
  float* kernel,
  float* target,
  shape_t kernel_s,
  shape_t image_s) {

  int mid_i = kernel_s.height / 2;
  int mid_j = kernel_s.width / 2;

  for (size_t i = 0; i < kernel_s.height; i++) {
    for (size_t j = 0; j < kernel_s.width; j++) {
      convolve_step(
        ipt,
        target,
        mat_get(kernel, kernel_s, i, j),
        i - mid_i,
        j - mid_j,
        image_s
      );
    }
  }
}

void convolve_step(
  float* ipt,
  float* target,
  float kernel_val,
  int kernel_offset_i,
  int kernel_offset_j,
  shape_t image_s) {

  for (size_t i = 0; i < image_s.height; i++) {
    int i2 = i + kernel_offset_i;

    if ((i2 < 0) || (i2 >= image_s.height)) {
      continue;
    }

    for (size_t j = 0; j < image_s.width; j++) {
      int j2 = j + kernel_offset_j;

      if ((j2 < 0) || (j2 >= image_s.width)) {
        continue;
      }

      *mat_offset(target, image_s, i, j) += \
        (mat_get(ipt, image_s, i2, j2) * kernel_val);
    }
  }
}

// In progress!
void convolve1d(
  float* ipt,
  float* kernel,
  float* des,
  shape_t image_s,
  int kernel_width) {

  assert(kernel_width == 3);

  float k0[8] = { [0 ... 7] = kernel[0] };
  float k1[8] = { [0 ... 7] = kernel[1] };
  float k2[8] = { [0 ... 7] = kernel[2] };

  __m256 k_buff0 = _mm256_loadu_ps(k0);
  __m256 k_buff1 = _mm256_loadu_ps(k1);
  __m256 k_buff2 = _mm256_loadu_ps(k2);

  __m256 data_buff;
  __m256 prod_buff0;
  __m256 prod_buff1;
  __m256 prod_buff2;
  __m256 result_buff;

  // int right_shift_cmd = 0x000000001010011100101110;
  // int left_shift_cmd =  0x001010011100101110111111;
  // int right_shift_arr[8] = { right_shift_cmd };
  // int left_shift_arr[8] = { left_shift_cmd };
  __m256i right_shift_buff = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
  __m256i left_shift_buff = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);

  for (size_t i = 0; i < image_s.height; i++) {
    for (size_t j = 0; j < image_s.width; j += 8) {
      data_buff = _mm256_loadu_ps(ipt + (i * image_s.width) + j);
      prod_buff0 = _mm256_mul_ps(k_buff0, data_buff);
      prod_buff1 = _mm256_mul_ps(k_buff1, data_buff);
      prod_buff2 = _mm256_mul_ps(k_buff2, data_buff);

      prod_buff0 = _mm256_permutevar8x32_ps(prod_buff0, right_shift_buff);
      prod_buff2 = _mm256_permutevar8x32_ps(prod_buff2, left_shift_buff);
    }
  }
}
