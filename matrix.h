typedef struct {
  int height;
  int width;
} shape_t;

#define HEIGHT 512
#define WIDTH 512
#define KSIZE 3
#define ITERS 1

float mat_get(float* matrix, shape_t shape, int i, int j);
void mat_set(float* matrix, shape_t shape, int i, int j, float val);
float* mat_offset(float* matrix, shape_t shape, int i, int j);

float* allocate_matrix(shape_t shape);
float* build_example_input(shape_t image_shape);
float* build_example_kernel(shape_t kernel_shape);

void print_matrix(float* mat, shape_t shape);
void print_matrix_corners(float* mat, shape_t shape);
