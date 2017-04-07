from avx_convolve2d cimport *
cimport cython
cimport numpy as np

cdef struct Result:
    float a0
    float a1
    float a2
    float a3

cdef shape_t image_shape
image_shape.height = 32
image_shape.width = 32
cdef shape_t kernel_shape
kernel_shape.height = 3
kernel_shape.width = 3

cdef float* ipt = build_example_input(image_shape)
cdef float* destination = allocate_matrix(image_shape)
cdef float* kernel = build_example_kernel(kernel_shape)

cdef Result result

cdef Result run_test_() nogil:
    convolve2d(ipt, kernel, destination, image_shape, kernel_shape)

    result.a0 = mat_get(destination, image_shape, 0, 0)
    result.a1 = mat_get(destination, image_shape, 0, 31)
    result.a2 = mat_get(destination, image_shape, 31, 0)
    result.a3 = mat_get(destination, image_shape, 31, 31)
    return result

def run_test():
    result = run_test_()
    return (result.a0, result.a1, result.a2, result.a3)

@cython.boundscheck(False)
cdef void avx_convolve2d_(
    np.float32_t[:, :] ipt,
    np.float32_t[:, :] kernel,
    np.float32_t[:, :] destination) nogil:

    cdef shape_t image_shape
    image_shape.height = ipt.shape[0]
    image_shape.width = ipt.shape[1]
    cdef shape_t kernel_shape
    kernel_shape.height = kernel.shape[0]
    kernel_shape.width = kernel.shape[1]

    convolve2d(
        &ipt[0, 0],
        &kernel[0, 0],
        &destination[0, 0],
        image_shape,
        kernel_shape
    )

@cython.boundscheck(False)
cdef void avx_backward_convolve2d_(
    np.float32_t[:, :] ipt,
    np.float32_t[:, :] kernel,
    np.float32_t[:, :] destination) nogil:

    cdef shape_t image_shape
    image_shape.height = ipt.shape[0]
    image_shape.width = ipt.shape[1]
    cdef shape_t kernel_shape
    kernel_shape.height = kernel.shape[0]
    kernel_shape.width = kernel.shape[1]

    backward_convolve2d(
        &ipt[0, 0],
        &kernel[0, 0],
        &destination[0, 0],
        image_shape,
        kernel_shape
    )

def avx_convolve2d(
    np.float32_t[:, :] ipt,
    np.float32_t[:, :] kernel,
    np.float32_t[:, :] destination):

    avx_convolve2d_(ipt, kernel, destination)
