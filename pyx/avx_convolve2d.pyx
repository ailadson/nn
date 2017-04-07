# This code translates from Numpy world to the AVX code we have
# written in C.

cimport avx_convolve2d_impl as avx_impl
cimport numpy as np
from tensor_utils cimport *

cdef enum direction_t:
    FORWARD,
    BACKWARD

# This method (1) iterates through the various layers, (2) calculates
# the proper offset into the matrices, and (3) passes this on to the
# AVX C code.
cdef void apply_convolution_(
    Rank3Tensor input_tensor,
    Rank4Tensor kernel_tensor,
    Rank3Tensor output_tensor,
    direction_t direction) nogil:

    cdef int input_layer_idx
    cdef int num_input_layers = input_tensor.shape.dim0
    cdef int output_layer_idx
    cdef int num_output_layers = output_tensor.shape.dim0

    cdef float* input_layer
    cdef float* kernel_layer
    cdef float* output_layer

    cdef avx_impl.shape_t image_shape
    image_shape.height = input_tensor.shape.dim1
    image_shape.width = input_tensor.shape.dim2
    cdef avx_impl.shape_t kernel_shape
    kernel_shape.height = kernel_tensor.shape.dim2
    kernel_shape.width = kernel_tensor.shape.dim3

    for output_layer_idx in range(num_output_layers):
        for input_layer_idx in range(num_input_layers):
            input_layer = rank3_offset(input_tensor, input_layer_idx)
            kernel_layer = rank4_offset(
                kernel_tensor, output_layer_idx, input_layer_idx
            )
            output_layer = rank3_offset(output_tensor, output_layer_idx)

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

# These are the final user-facing methods!
def apply_convolution(
    DTYPE_t[:, :, :] input_layers,
    DTYPE_t[:, :, :, :] kernel_layers,
    DTYPE_t[:, :, :] output_layers):

    cdef Rank3Tensor input_tensor = memview_to_rank3_tensor(
        input_layers
    )
    cdef Rank4Tensor kernel_tensor = memview_to_rank4_tensor(
        kernel_layers
    )
    cdef Rank3Tensor output_tensor = memview_to_rank3_tensor(
        output_layers
    )

    apply_convolution_(
        input_tensor, kernel_tensor, output_tensor, FORWARD
    )

def apply_backward_convolution(
    DTYPE_t[:, :, :] input_layers,
    DTYPE_t[:, :, :, :] kernel_layers,
    DTYPE_t[:, :, :] output_layers):

    cdef Rank3Tensor input_tensor = memview_to_rank3_tensor(
        input_layers
    )
    cdef Rank4Tensor kernel_tensor = memview_to_rank4_tensor(
        kernel_layers
    )
    cdef Rank3Tensor output_tensor = memview_to_rank3_tensor(
        output_layers
    )

    apply_convolution_(
        input_tensor, kernel_tensor, output_tensor, BACKWARD
    )
