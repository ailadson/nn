cimport numpy as np

ctypedef np.float32_t DTYPE_t

cdef struct Rank2Shape:
    int dim0
    int dim1

cdef struct Rank2Tensor:
    DTYPE_t* data
    Rank2Shape shape

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

cdef inline DTYPE_t rank2_get(Rank2Tensor t, size_t i, size_t j) nogil:
    return (t.data + (i * t.shape.dim0) + j)[0]

cdef inline void rank2_inc(
    Rank2Tensor t, size_t i, size_t j, DTYPE_t val) nogil:
    (t.data + (i * t.shape.dim0) + j)[0] += val

# Helpers for calculation of offsets.
cdef inline Rank2Tensor rank3_offset(
    Rank3Tensor in_tensor, size_t i) nogil:

    cdef Rank2Tensor out_tensor
    out_tensor.data = in_tensor.data
    out_tensor.data += i * in_tensor.shape.dim1 * in_tensor.shape.dim2
    out_tensor.shape.dim0 = in_tensor.shape.dim1
    out_tensor.shape.dim1 = in_tensor.shape.dim2

    return out_tensor

cdef inline Rank2Tensor rank4_offset(
    Rank4Tensor in_tensor,
    size_t i,
    size_t j) nogil:

    cdef Rank2Tensor out_tensor
    out_tensor.data = in_tensor.data
    out_tensor.data += (
        i * in_tensor.shape.dim1 * in_tensor.shape.dim2 * in_tensor.shape.dim3
    )
    out_tensor.data += j * in_tensor.shape.dim2 * in_tensor.shape.dim3
    out_tensor.shape.dim0 = in_tensor.shape.dim2
    out_tensor.shape.dim1 = in_tensor.shape.dim3

    return out_tensor

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
