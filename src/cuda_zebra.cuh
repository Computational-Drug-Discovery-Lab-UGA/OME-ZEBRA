#ifndef CUDA_ZEBRA_CUH
#define CUDA_ZEBRA_CUH

#include "common_includes.h"
#include <sys/time.h>
#include <math.h>
#include "matrix.cuh"

__global__ void findMinMax(uint32* mtx, unsigned long size, uint32* min, uint32* max);
__global__ void normalize(uint32 *mtx, float *normals, uint32* min, uint32* max, unsigned long size);
__global__ void generateKey(unsigned long numPixels, unsigned int numTimePoints, float* mtx, bool* key);
__global__ void randInitMatrix(unsigned long size, float* mtx);
__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, long diffDimA,
   long comDim, long diffDimB);

void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block);
void getGrid(unsigned long size, dim3 &grid, int blockSize);
float* executeNormalization(uint32* mtx, unsigned long size);
bool* generateKey(unsigned long numPixels, unsigned int numTimePoints, float* mtx, unsigned long &numPixelsWithValues);
float* minimizeVideo(unsigned long numPixels, unsigned long numPixelsWithValues, unsigned int numTimePoints, float* mtx, bool* key);

double get_time();
int start_time(double* t, int i);
int stop_time(double* t, int i);
unsigned nextpow2(unsigned x);
void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double* t, int verbose);

void performNNMF(float* &W, float* &H, float* V, unsigned int k, unsigned long numPixels, unsigned int numTimePoints);

#endif /* CUDA_ZEBRA_CUH */
