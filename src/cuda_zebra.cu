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

__global__ void findMinMax(uint32* mtx, unsigned long size, uint32* min, uint32* max){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  __shared__ uint32 bmax;
  __shared__ uint32 bmin;
  bmax = 0;
  bmin = UINT32_MAX;
  __syncthreads();
  if(globalID < size){
    uint32 value = mtx[globalID];
    if(value != 0){
      atomicMax(&bmax, value);
      atomicMin(&bmin, value);
    }
  }
  __syncthreads();
  if(threadIdx.x == 0){
    atomicMax(max, bmax);
    atomicMin(min, bmin);
  }
}
__global__ void normalize(uint32 *mtx, float *normals, uint32* min, uint32* max, unsigned long size) {
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float currentValue = 0;
  float dmin = static_cast<float>(*min);
  float dmax = static_cast<float>(*max);
  while(globalID < size){
    if (mtx[globalID] != 0) {
      currentValue = static_cast<float>(mtx[globalID]) - dmin;
      currentValue /= (dmax - dmin);
    }
    normals[globalID] = currentValue;
    normals[globalID] = 1.0f / (1.0f + expf((-10.0f * currentValue) + 7.5));
    //printf("%f\n",normals[globalID]);
    globalID += stride;
  }
}
__global__ void generateKey(unsigned long numPixels, unsigned int numTimePoints, float* mtx, bool* key){
  long blockID = blockIdx.y * gridDim.x + blockIdx.x;
  if(blockID < numPixels){
    __shared__ bool hasNonZero;
    hasNonZero = false;
    __syncthreads();
    for(int tp = threadIdx.x; tp < numTimePoints; tp += blockDim.x){
      if(hasNonZero) return;
      if(mtx[blockID*numTimePoints + tp] != 0.0f){
        key[blockID] = true;
        hasNonZero = true;
        return;
      }
    }
    __syncthreads();
    if(!hasNonZero){
      key[blockID] = false;
      return;
    }
  }
}
__global__ void randInitMatrix(unsigned long size, float* mtx){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  if(globalID < size){
    mtx[globalID] = ((float)(clock64()%1000))/1000.0f;
    if(mtx[globalID] == 0.0f) mtx[globalID] += 2e-30;
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

void getFlatGridBlock(unsigned long size, dim3 &grid, dim3 &block) {
  if(2147483647 > size){
    grid.x = size;
  }
  else if((unsigned long) 2147483647 * 1024 > size){
    grid.x = 2147483647;
    block.x = 1024;
    while(block.x * grid.x > size){
      block.x--;
    }
    block.x++;
  }
  else{
    grid.x = 65535;
    block.x = 1024;
    grid.y = 1;
    while(grid.x * grid.y * block.x < size){
      grid.y++;
    }
  }
}
void getGrid(unsigned long size, dim3 &grid, int blockSize) {
  if(2147483647 > size){
    grid.x = size;
  }
  else{
    grid.x = 65535;
    grid.y = 1;
    while(grid.x * grid.y * grid.y < size){
      grid.y++;
    }
  }
}
float* executeNormalization(uint32* mtx, unsigned long size){
  uint32 max = 0;
  uint32 min = UINT32_MAX;
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(size, grid, block);

  float* norm = new float[size];
  uint32* maxd;
  uint32* mind;
  uint32* matrixDevice;
  float* normDevice;
  CudaSafeCall(cudaMalloc((void**)&maxd, sizeof(uint32)));
  CudaSafeCall(cudaMalloc((void**)&mind, sizeof(uint32)));
  CudaSafeCall(cudaMalloc((void**)&matrixDevice, size*sizeof(uint32)));
  CudaSafeCall(cudaMalloc((void**)&normDevice, size*sizeof(float)));
  CudaSafeCall(cudaMemcpy(maxd, &max, sizeof(uint32), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(mind, &min, sizeof(uint32), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(matrixDevice, mtx, size*sizeof(uint32), cudaMemcpyHostToDevice));

  std::cout<<"searching for max and min"<<std::endl;
  findMinMax<<<grid,block>>>(matrixDevice, size, mind, maxd);
  cudaDeviceSynchronize();
  CudaCheckError();
  std::cout<<"executing normalization"<<std::endl;
  normalize<<<grid,block>>>(matrixDevice, normDevice, mind, maxd, size);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(&max, maxd, sizeof(uint32), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(&min, mind, sizeof(uint32), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(norm, normDevice, size*sizeof(float), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(maxd));
  CudaSafeCall(cudaFree(mind));
  CudaSafeCall(cudaFree(matrixDevice));
  CudaSafeCall(cudaFree(normDevice));
  printf("whole video - (uint32) min = %d, max = %d\n",min,max);
  return norm;

}
bool* generateKey(unsigned long numPixels, unsigned int numTimePoints, float* mtx, unsigned long &numPixelsWithValues){
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  block.x = (numTimePoints < 1024) ? numTimePoints : 1024;
  getGrid(numPixels, grid, block.x);

  bool* key = new bool[numPixels];

  float* matrixDevice;
  bool* keyDevice;

  CudaSafeCall(cudaMalloc((void**)&matrixDevice, numPixels*numTimePoints*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&keyDevice, numPixels*sizeof(float)));
  CudaSafeCall(cudaMemcpy(matrixDevice, mtx, numPixels*numTimePoints*sizeof(float), cudaMemcpyHostToDevice));
  std::cout<<"generating key to eradicate pixels that are always 0 = ";

  generateKey<<<grid,block>>>(numPixels, numTimePoints, matrixDevice, keyDevice);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(key, keyDevice, numPixels*sizeof(bool), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(matrixDevice));
  CudaSafeCall(cudaFree(keyDevice));
  for(int p = 0; p < numPixels; ++p){
    if(key[p]) ++numPixelsWithValues;
  }
  std::cout<<numPixels - numPixelsWithValues<<std::endl;

  return key;

}
float* minimizeVideo(unsigned long numPixels, unsigned long numPixelsWithValues, unsigned int numTimePoints, float* mtx, bool* key){
  std::cout<<"minimizing video due existence of all 0 rows"<<std::endl;
  float* minimizedVideo = new float[numPixelsWithValues*numTimePoints];
  int currentPixel = 0;
  for(int p = 0; p < numPixels; ++p){
    if(key[p]){
      memcpy(&minimizedVideo[currentPixel*numTimePoints], mtx + p*numTimePoints, numTimePoints*sizeof(float));
      ++currentPixel;
    }
  }
  return minimizedVideo;
}

void performNNMF(float* &W, float* &H, float* V, unsigned int k, unsigned long numPixels, unsigned int numTimePoints){
  float* dW;
  float* dH;

  CudaSafeCall(cudaMalloc((void**)&dW, numPixels*k*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&dH, k*numTimePoints*sizeof(float)));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numPixels*k, grid, block);
  randInitMatrix<<<grid,block>>>(numPixels*k, dW);
  CudaCheckError();
  grid = {1,1,1};
  block = {1,1,1};
  getFlatGridBlock(k*numTimePoints, grid, block);
  randInitMatrix<<<grid,block>>>(k*numTimePoints, dH);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(W, dW, numPixels*k*sizeof(float), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(H, dH, k*numTimePoints*sizeof(float), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(dW));
  CudaSafeCall(cudaFree(dH));

  clock_t nnmfTimer;
  nnmfTimer = clock();
  std::cout<<"starting nnmf"<<std::endl;
  printf("%f,%f\n",W[0],H[0]);

  /*DO NMF*/


  printf("nnmf took %f seconds.\n\n", ((float) clock() - nnmfTimer)/CLOCKS_PER_SEC);
}
