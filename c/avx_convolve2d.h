#include "matrix.h"

void convolve2d(
  float* input,
  float* kernel,
  float* destination,
  shape_t image_shape,
  shape_t kernel_shape
);

void backward_convolve2d(
  float* input,
  float* kernel,
  float* destination,
  shape_t image_shape,
  shape_t kernel_shape
);
