#ifndef CUDA_ZEBRA_CUH
#define CUDA_ZEBRA_CUH

#include "common_includes.h"
#include "Python.h"
#include "/usr/lib/python3/dist-packages/numpy/core/include/numpy/arrayobject.h"

__device__ __forceinline__ int floatToOrderedInt(float floatVal);
__device__ __forceinline__ float orderedIntToFloat(int intVal);
__global__ void ensurePositivity(float* mtx, unsigned long size, int* globalPlaceHolder);
__global__ void findMinMax(uint32* mtx, unsigned long size, uint32* min, uint32* max);
__global__ void normalize(uint32 *mtx, float *normals, uint32* min, uint32* max, unsigned long size);
__global__ void floatToUINT32(float *mtx, float min, float max, unsigned long size);
__global__ void generateKey(unsigned long numPixels, unsigned int numTimePoints, float* mtx, bool* key);
__global__ void randInitMatrix(unsigned long size, float* mtx);
__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, long diffDimA, long comDim, long diffDimB);
__global__ void multiplyMatrices(float *matrixA, float *matrixB, uint32 *resultTranspose, long diffDimA, long comDim, long diffDimB);
void executeMultiplyMatrices(float *matrixA, float *matrixB, float* &matrixC, long diffDimA, long comDim, long diffDimB);

void executeMultiplyMatrices(float *matrixA, float *matrixB, float* &matrixC,long diffDimA, long comDim, long diffDimB);
void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block);
void getGrid(unsigned long size, dim3 &grid, int blockSize);
float* executeNormalization(uint32* mtx, unsigned long size);
bool* generateKey(unsigned long numPixels, unsigned int numTimePoints, float* mtx, unsigned long &numPixelsWithValues);
float* minimizeVideo(unsigned long numPixels, unsigned long numPixelsWithValues, unsigned int numTimePoints, float* mtx, bool* key);

void performNNMF(float* &W, float* &H, float* V, unsigned int k, unsigned long numPixels, unsigned int numTimePoints, std::string outDir);

#endif /* CUDA_ZEBRA_CUH */
