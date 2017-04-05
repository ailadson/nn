from convolve2d import convolve2d
import numpy as np
import time

WIDTH = 28
HEIGHT = 28
KSIZE = 3
ITERS = 100000

ipt = np.zeros([HEIGHT, WIDTH])
des = np.zeros([HEIGHT, WIDTH])
kernel = np.zeros([KSIZE, KSIZE])

t = time.time()
for i in range(ITERS):
    convolve2d(ipt, kernel, des)
print(time.time() - t)

#   #include <immintrin.h>
#
# int main(){
#   __m256 ymm0, ymm1;//define the registers used
#   float a[8] = {1,2,3,4,5,6,7,8};
#   float b[8]={2,3,4,5,6,7,8,9};
#   float c[8];
#   ymm0 = __builtin_ia32_loadups256(a);//load the 8 floats in a into ymm0
#   ymm1 = __builtin_ia32_loadups256(b);//load the 8 floats in b into ymm1
# 
#   //multiply ymm0 and ymm1, store the result in ymm0
#   ymm0 = __builtin_ia32_mulps256(ymm0, ymm1);
#   __builtin_ia32_storeups256(c, ymm0);//copy the 8 floats in ymm0 to c
#
#   for (int i=0; i<8; i++)
#     std::cout<<c[i]<<", ";
#   return 0;
# }
