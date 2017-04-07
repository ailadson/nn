#include <assert.h>
#include "avx_convolve2d.h"
#include <immintrin.h>
#include "matrix.h"
#include <string.h>

void convolve1d(float* input,
                float* kernel,
                float* destination,
                shape_t image_shape,
                size_t kernel_width,
                size_t kernel_row_offset) {
  assert(kernel_width == 3);

  float k0_arr[8] = { [0 ... 7] = kernel[0] };
  float k1_arr[8] = { [0 ... 7] = kernel[1] };
  float k2_arr[8] = { [0 ... 7] = kernel[2] };
  __m256 k0_avx = _mm256_loadu_ps(k0_arr);
  __m256 k1_avx = _mm256_loadu_ps(k1_arr);
  __m256 k2_avx = _mm256_loadu_ps(k2_arr);

  // Hack to get a float of zeros and a float of ones.
  float zeros_float;
  float ones_float;
  memset((char*) &zeros_float, 0, sizeof(float));
  memset((char*) &ones_float, 0xffffffff, sizeof(float));

  __m256i right_shift_avx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
  float drop_left_el_arr[] = { zeros_float, [1 ... 7] = ones_float };
  __m256 drop_left_el_avx = _mm256_loadu_ps(drop_left_el_arr);

  __m256i left_shift_avx = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
  float drop_right_el_arr[] = { [0 ... 6] = ones_float, zeros_float };
  __m256 drop_right_el_avx = _mm256_loadu_ps(drop_right_el_arr);

  __m256 data_avx;
  __m256 result_avx;
  __m256 prod0_avx;
  __m256 prod1_avx;
  __m256 prod2_avx;

  // This approach avoids memcpy/memset and does AVX masking. But it
  // relies on you having allocated memory in blocks of
  // 8*sizeof(float), because it does slip over the array a little
  // bit.
  float mask_row_overhang_arr[8];
  memset((char*) mask_row_overhang_arr, 0x00, 8 * sizeof(float));
  memset((char*) mask_row_overhang_arr,
         0xffffffff,
         (image_shape.width % 8) * sizeof(float));
  __m256 mask_row_overhang_avx = _mm256_loadu_ps(mask_row_overhang_arr);

  for (size_t i = 0; i < image_shape.height; i++) {
    // Notice that we *subtract* the kernel_row_offset. This is
    // correct.
    size_t destination_row_idx = i - kernel_row_offset;
    if (!row_in_bounds(image_shape, destination_row_idx)) {
      continue;
    }

    for (size_t j = 0; j < image_shape.width; j += 8) {
      float* read_addr = mat_offset(input, image_shape, i, j);

      data_avx = _mm256_loadu_ps(read_addr);
      if ((j + 7) >= image_shape.width) {
        data_avx = _mm256_and_ps(data_avx, mask_row_overhang_avx);
      }

      prod0_avx = _mm256_mul_ps(k0_avx, data_avx);
      prod1_avx = _mm256_mul_ps(k1_avx, data_avx);
      prod2_avx = _mm256_mul_ps(k2_avx, data_avx);

      prod0_avx = _mm256_permutevar8x32_ps(prod0_avx, right_shift_avx);
      prod0_avx = _mm256_and_ps(prod0_avx, drop_left_el_avx);
      prod2_avx = _mm256_permutevar8x32_ps(prod2_avx, left_shift_avx);
      prod2_avx = _mm256_and_ps(prod2_avx, drop_right_el_avx);

      result_avx = _mm256_add_ps(prod0_avx, prod1_avx);
      result_avx = _mm256_add_ps(result_avx, prod2_avx);
      if ((j + 7) >= image_shape.width) {
        result_avx = _mm256_and_ps(result_avx, mask_row_overhang_avx);
      }

      // Now load the destination and increment.
      float* destination_addr = \
        mat_offset(destination, image_shape, destination_row_idx, j);
      __m256 destination_avx = _mm256_loadu_ps(destination_addr);
      result_avx = _mm256_add_ps(result_avx, destination_avx);
      _mm256_storeu_ps(destination_addr, result_avx);

      // Don't forget the folks at the ends!
      if (j > 0) {
        destination_addr[0] += \
          kernel[0] * mat_get(input, image_shape, i, j - 1);
      }
      if ((j+7) < (image_shape.width - 1)) {
        destination_addr[7] += \
          kernel[2] * mat_get(input, image_shape, i, j + 8);
      }
    }
  }
}

void convolve2d(float* input,
                float* kernel,
                float* destination,
                shape_t image_shape,
                shape_t kernel_shape) {
  size_t mid_i = kernel_shape.height / 2;
  for (size_t i = 0; i < kernel_shape.height; i++) {
    size_t kernel_row_offset = i - mid_i;
    convolve1d(input,
               mat_offset(kernel, kernel_shape, i, 0),
               destination,
               image_shape,
               kernel_shape.width,
               kernel_row_offset);
  }
}
