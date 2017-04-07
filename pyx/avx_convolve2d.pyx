# This code translates from Numpy world to the AVX code we have
# written in C.

cimport avx_convolve2d_impl as avx_impl
cimport numpy as np

ctypedef np.float32_t DTYPE_t

cdef enum direction_t:
    FORWARD,
    BACKWARD

# These structs are an alternative to the shape attribute.
cdef struct Rank2Shape:
    int dim0
    int dim1

cdef struct Rank3Shape:
    int dim0
    int dim1
    int dim2

cdef struct Rank4Shape:
    int dim0
    int dim1
    int dim2
    int dim3

# Helpers for calculation of offsets.
cdef float* rank3_offset(
    float* matrix,
    Rank3Shape shape,
    size_t i) nogil:

    return matrix + i * (shape.dim1 * shape.dim2)

cdef float* rank4_offset(
    float* matrix,
    Rank4Shape shape,
    size_t i,
    size_t j) nogil:

    cdef float* result = matrix
    result += i * (shape.dim1 * shape.dim2 * shape.dim3)
    result += j * (shape.dim2 * shape.dim3)
    return result

# This method (1) iterates through the various layers, (2) calculates
# the proper offset into the matrices, and (3) passes this on to the
# AVX C code.
cdef void apply_convolution2_(
    DTYPE_t* input_layers,
    Rank3Shape input_layers_shape,
    DTYPE_t* kernel_layers,
    Rank4Shape kernel_layers_shape,
    DTYPE_t* output_layers,
    Rank3Shape output_layers_shape,
    direction_t direction) nogil:

    cdef int input_layer_idx
    cdef int num_input_layers = input_layers_shape.dim0
    cdef int output_layer_idx
    cdef int num_output_layers = output_layers_shape.dim0

    cdef float* input_layer
    cdef float* kernel_layer
    cdef float* output_layer

    cdef avx_impl.shape_t image_shape
    image_shape.height = input_layers_shape.dim1
    image_shape.width = input_layers_shape.dim2
    cdef avx_impl.shape_t kernel_shape
    kernel_shape.height = kernel_layers_shape.dim2
    kernel_shape.width = kernel_layers_shape.dim3

    for output_layer_idx in range(num_output_layers):
        for input_layer_idx in range(num_input_layers):
            input_layer = rank3_offset(
                input_layers, input_layers_shape, input_layer_idx
            )

            kernel_layer = rank4_offset(
                kernel_layers,
                kernel_layers_shape,
                output_layer_idx,
                input_layer_idx
            )
            output_layer = rank3_offset(
                output_layers, output_layers_shape, output_layer_idx
            )

            if (direction == FORWARD):
                avx_impl.convolve2d(
                    input_layer,
                    kernel_layer,
                    output_layer,
                    image_shape,
                    kernel_shape
                )
            else:
                avx_impl.backward_convolve2d(
                    input_layer,
                    kernel_layer,
                    output_layer,
                    image_shape,
                    kernel_shape
                )

# This method prepares a bunch of pointers and shape objects.
cdef void apply_convolution_(
    DTYPE_t[:, :, :] input_layers,
    DTYPE_t[:, :, :, :] kernel_layers,
    DTYPE_t[:, :, :] output_layers,
    direction_t direction) nogil:

    cdef Rank3Shape input_layers_shape
    input_layers_shape.dim0 = input_layers.shape[0]
    input_layers_shape.dim1 = input_layers.shape[1]
    input_layers_shape.dim2 = input_layers.shape[2]
    cdef Rank4Shape kernel_layers_shape
    kernel_layers_shape.dim0 = kernel_layers.shape[0]
    kernel_layers_shape.dim1 = kernel_layers.shape[1]
    kernel_layers_shape.dim2 = kernel_layers.shape[2]
    kernel_layers_shape.dim3 = kernel_layers.shape[3]
    cdef Rank3Shape output_layers_shape
    output_layers_shape.dim0 = output_layers.shape[0]
    output_layers_shape.dim1 = output_layers.shape[1]
    output_layers_shape.dim2 = output_layers.shape[2]

    apply_convolution2_(
        &input_layers[0, 0, 0],
        input_layers_shape,
        &kernel_layers[0, 0, 0, 0],
        kernel_layers_shape,
        &output_layers[0, 0, 0],
        output_layers_shape,
        direction
    )

# These are the final user-facing methods!
def apply_convolution(
    DTYPE_t[:, :, :] input_layers,
    DTYPE_t[:, :, :, :] kernel_layers,
    DTYPE_t[:, :, :] output_layers):
    apply_convolution_(
        input_layers, kernel_layers, output_layers, FORWARD
    )

def apply_backward_convolution(
    DTYPE_t[:, :, :] input_layers,
    DTYPE_t[:, :, :, :] kernel_layers,
    DTYPE_t[:, :, :] output_layers):
    apply_convolution_(
        input_layers, kernel_layers, output_layers, BACKWARD
    )
