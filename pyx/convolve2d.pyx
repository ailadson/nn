cimport avx_convolve2d as avx
cimport cython
cimport numpy as np

ctypedef np.float32_t DTYPE_t

cdef enum direction_t:
    FORWARD,
    BACKWARD

# === Traditional Cython Implementation ===
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void convolve_step(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] target,
    DTYPE_t kval,
    int kernel_offset_i,
    int kernel_offset_j) nogil:

    cdef int num_rows = ipt.shape[0]
    cdef int num_cols = ipt.shape[1]
    cdef int i, j
    cdef int i2, j2

    for i in range(num_rows):
        for j in range(num_cols):
            i2 = i + kernel_offset_i
            j2 = j + kernel_offset_j

            if (i2 < 0) | (i2 > num_rows):
                continue
            if (j2 < 0) | (j2 > num_rows):
                continue

            target[i, j] += ipt[i2, j2] * kval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void convolve2d_(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] kernel,
    DTYPE_t[:, :] target) nogil:

    cdef int krows = kernel.shape[0]
    cdef int kcols = kernel.shape[1]
    cdef int mid_i = kernel.shape[0] // 2
    cdef int mid_j = kernel.shape[1] // 2
    cdef int i, j

    for i in range(krows):
        for j in range(kcols):
            convolve_step(
                ipt, target, kernel[i, j], i - mid_i, j - mid_j
            )

@cython.boundscheck(False)
cdef void backward_convolve2d_(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] kernel,
    DTYPE_t[:, :] target) nogil:

    cdef int krows = kernel.shape[0]
    cdef int kcols = kernel.shape[1]
    # If the kernel has a non-odd KSIZE I believe the center is
    # adjusted slightly... This code may be wrong though.
    cdef int mid_i = (krows - 1) - (krows // 2)
    cdef int mid_j = (kcols - 1) - (kcols // 2)
    cdef int i, j

    for i in range(krows):
        for j in range(kcols):
            # Backward convolution means flipping the kernel
            # left/right and up/down.
            i = (krows - 1) - i
            j = (kcols - 1) - j
            convolve_step(
                ipt, target, kernel[i, j], i - mid_i, j - mid_j
            )

def convolve2d(
    DTYPE_t[:, :] ipt,
    DTYPE_t[:, :] kernel,
    DTYPE_t[:, :] target):

    convolve2d_(ipt, kernel, target)

# === AVX Implementation ===

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

    cdef avx.shape_t image_shape
    image_shape.height = input_layers_shape.dim1
    image_shape.width = input_layers_shape.dim2
    cdef avx.shape_t kernel_shape
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
                avx.convolve2d(
                    input_layer,
                    kernel_layer,
                    output_layer,
                    image_shape,
                    kernel_shape
                )
            else:
                avx.backward_convolve2d(
                    input_layer,
                    kernel_layer,
                    output_layer,
                    image_shape,
                    kernel_shape
                )

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
