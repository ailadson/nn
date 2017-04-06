#include "avx_convolve2d.h"
#include "matrix.h"
#include <stdio.h>
#include <time.h>

int main() {
  shape_t image_shape = { .height = HEIGHT, .width = WIDTH };
  float* input = build_example_input(image_shape);
  float* destination = allocate_matrix(image_shape);
  shape_t kernel_shape = { .height = KSIZE, .width = KSIZE };
  float* kernel = build_example_kernel(kernel_shape);

  printf("Input matrix!\n");
  print_matrix_corners(input, image_shape);
  printf("Kernel matrix!\n");
  print_matrix(kernel, kernel_shape);

  long t = clock();
  for (size_t i = 0; i < ITERS; i++) {
    convolve2d(input,
               kernel,
               destination,
               image_shape,
               kernel_shape);
  }
  t = clock() - t;

  printf("Result matrix!\n");
  print_matrix_corners(destination, image_shape);

  printf("time: %f\n", (double) t / CLOCKS_PER_SEC);

  return 0;
}
