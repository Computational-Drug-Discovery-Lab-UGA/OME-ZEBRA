#include "cuda_zebra.cuh"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  // err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}


__global__ void normalize(uint32 *flatMatrix, float *normals, uint32 min, uint32 max, long size) {
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float currentValue = 0;
  float dmin = static_cast<float>(min);
  float dmax = static_cast<float>(max);
  while(globalID < size){
    if (flatMatrix[globalID] != 0) {
      currentValue = static_cast<float>(flatMatrix[globalID]) - dmin;
      currentValue /= (dmax - dmin);
    }
    normals[globalID] = 1.0f / (1.0f + expf((-10.0f * currentValue) + 7.5));
    globalID += stride;
  }
}
__global__ void multiplyMatrices(float *matrixA, float *matrixB, float *matrixC,
                                 long diffDimA, long comDim, long diffDimB){

  long blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  long currentIndex = globalID;

  if(currentIndex < (diffDimA * diffDimB)){

    long iIndex = currentIndex / diffDimB;
    long jIndex = currentIndex % diffDimB;

    float sum = 0;

    for(int k = 0; k < comDim; k++){

      sum += (matrixA[iIndex * comDim + k] * matrixB[k * diffDimB + jIndex]);
    }

    matrixC[iIndex * diffDimB + jIndex] = sum;
  }
}
