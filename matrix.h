#include <stdbool.h>

typedef struct {
  int height;
  int width;
} shape_t;

#define HEIGHT 24
#define WIDTH 24
#define KSIZE 3
#define ITERS (100 * 1000)

float mat_get(float* matrix, shape_t shape, int i, int j);
void mat_set(float* matrix, shape_t shape, int i, int j, float val);
float* mat_offset(float* matrix, shape_t shape, int i, int j);

bool row_in_bounds(shape_t shape, int row_idx);

float* allocate_matrix(shape_t shape);
float* build_example_input(shape_t image_shape);
float* build_example_kernel(shape_t kernel_shape);

void print_matrix(float* mat, shape_t shape);
void print_matrix_corners(float* mat, shape_t shape);
