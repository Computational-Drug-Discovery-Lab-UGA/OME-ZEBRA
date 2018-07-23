#ifndef CUDA_ZEBRA_H
#define CUDA_ZEBRA_H

#include "common_includes.h"

__global__ void normalize(uint32 *flatMatrix, float *normals, uint32 min, uint32 max, long size);
__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, long diffDimA,
   long comDim, long diffDimB);

#endif /* CUDA_ZEBRA_H */
