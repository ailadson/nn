cdef extern from "avx_convolve2d.h":
    void convolve2d(float* input,
                    float* kernel,
                    float* destination,
                    shape_t image_shape,
                    shape_t kernel_shape) nogil;

cdef extern from "matrix.h":
    ctypedef struct shape_t:
        size_t height
        size_t width

    float* allocate_matrix(shape_t shape) nogil;
    float* build_example_input(shape_t image_shape) nogil;
    float* build_example_kernel(shape_t kernel_shape) nogil;

    float mat_get(float*, shape_t, size_t, size_t) nogil;
