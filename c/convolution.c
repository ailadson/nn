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
  shape_t kernel_s = { .width = KSIZE, .height = KSIZE };
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
  size_t kernel_offset_i,
  size_t kernel_offset_j,
  shape_t image_s);

void convolve2d(
  float* ipt,
  float* kernel,
  float* target,
  shape_t kernel_s,
  shape_t image_s) {

  size_t mid_i = kernel_s.height / 2;
  size_t mid_j = kernel_s.width / 2;

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
  size_t kernel_offset_i,
  size_t kernel_offset_j,
  shape_t image_s) {

  for (size_t i = 0; i < image_s.height; i++) {
    size_t i2 = i + kernel_offset_i;

    if (!row_in_bounds(image_s, i2)) {
      continue;
    }

    for (size_t j = 0; j < image_s.width; j++) {
      size_t j2 = j + kernel_offset_j;

      if (!col_in_bounds(image_s, j2)) {
        continue;
      }

      *mat_offset(target, image_s, i, j) += \
        (mat_get(ipt, image_s, i2, j2) * kernel_val);
    }
  }
}
