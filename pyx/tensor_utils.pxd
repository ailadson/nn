cimport numpy as np

ctypedef np.float32_t DTYPE_t

cdef struct Rank2Shape:
    int dim0
    int dim1

cdef struct Rank3Shape:
    int dim0
    int dim1
    int dim2

cdef struct Rank3Tensor:
    DTYPE_t* data
    Rank3Shape shape

cdef struct Rank4Shape:
    int dim0
    int dim1
    int dim2
    int dim3

cdef struct Rank4Tensor:
    DTYPE_t* data
    Rank4Shape shape

# Helpers for calculation of offsets.
cdef inline DTYPE_t* rank3_offset(Rank3Tensor t, size_t i) nogil:
    return t.data + i * (t.shape.dim1 * t.shape.dim2)

cdef inline DTYPE_t* rank4_offset(
    Rank4Tensor t,
    size_t i,
    size_t j) nogil:

    cdef DTYPE_t* result = t.data
    result += i * (t.shape.dim1 * t.shape.dim2 * t.shape.dim3)
    result += j * (t.shape.dim2 * t.shape.dim3)
    return result

cdef inline Rank3Tensor memview_to_rank3_tensor(
    DTYPE_t[:, :, :] x) nogil:

    cdef Rank3Tensor tensor
    tensor.data = &x[0, 0, 0]
    tensor.shape.dim0 = x.shape[0]
    tensor.shape.dim1 = x.shape[1]
    tensor.shape.dim2 = x.shape[2]

    return tensor

cdef inline Rank4Tensor memview_to_rank4_tensor(
    DTYPE_t[:, :, :, :] x) nogil:

    cdef Rank4Tensor tensor
    tensor.data = &x[0, 0, 0, 0]
    tensor.shape.dim0 = x.shape[0]
    tensor.shape.dim1 = x.shape[1]
    tensor.shape.dim2 = x.shape[2]
    tensor.shape.dim3 = x.shape[3]

    return tensor
