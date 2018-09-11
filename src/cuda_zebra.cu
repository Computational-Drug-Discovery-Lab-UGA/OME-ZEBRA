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

__device__ __forceinline__ int floatToOrderedInt(float floatVal){
 int intVal = __float_as_int( floatVal );
 return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float orderedIntToFloat(int intVal){
 return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

__global__ void ensurePositivity(float* mtx, unsigned long size, int* globalPlaceHolder){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  if(globalID < size){
    atomicMin(globalPlaceHolder, floatToOrderedInt(mtx[globalID]));
    cudaDeviceSynchronize();
    mtx[globalID] -= (orderedIntToFloat(*globalPlaceHolder) - 0.1);
  }
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
__global__ void floatToUINT32(float *mtx, float min, float max, unsigned long size) {
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float currentValue = 0;
  float regMin = min;
  float regMax = max;
  float maxUINT32 = UINT32_MAX;
  while(globalID < size){
    if (mtx[globalID] != 0) {
      currentValue = mtx[globalID] - regMin;
      currentValue /= (regMax - regMin);
    }
    mtx[globalID] = (currentValue*maxUINT32);
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
  }
}
__global__ void multiplyMatrices(float *matrixA, float *matrixB, float *matrixC, long diffDimA, long comDim, long diffDimB){

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
__global__ void multiplyMatrices(float *matrixA, float *matrixB, uint32 *resultTranspose, long diffDimA, long comDim, long diffDimB){

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

    //result[iIndex * diffDimB + jIndex] = __float_as_uint(sum);
    if(sum == 0.0f) printf("YO\n");
    resultTranspose[jIndex * diffDimA + iIndex] = floatToOrderedInt(sum);

  }
}
void executeMultiplyMatrices(float *matrixA, float *matrixB, float* &matrixC, long diffDimA, long comDim, long diffDimB){

  float* matrixADevice, *matrixBDevice, *matrixCDevice;

  CudaSafeCall(cudaMalloc((void**)&matrixADevice, diffDimA*comDim*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&matrixBDevice, comDim*diffDimB*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&matrixCDevice, diffDimA*diffDimB*sizeof(float)));

  CudaSafeCall(cudaMemcpy(matrixADevice, matrixA, diffDimA*comDim*sizeof(float), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(matrixBDevice, matrixB, comDim*diffDimB*sizeof(float), cudaMemcpyHostToDevice));

  dim3 grid, block;

  getFlatGridBlock(diffDimA*diffDimB, grid, block);

  multiplyMatrices<<<grid, block>>>(matrixADevice, matrixBDevice, matrixCDevice, diffDimA, comDim, diffDimB);

  CudaSafeCall(cudaMemcpy(matrixC, matrixCDevice, diffDimA*diffDimB*sizeof(float), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(matrixADevice));
  CudaSafeCall(cudaFree(matrixBDevice));
  CudaSafeCall(cudaFree(matrixCDevice));

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

void performNNMF(float* &W, float* &H, float* V, unsigned int k, unsigned long numPixels, unsigned int numTimePoints, std::string baseDir){
  clock_t nnmfTimer;
  nnmfTimer = clock();
  std::cout<<"starting nnmf"<<std::endl;
  std::cout<<"ensuring positivity"<<std::endl;
  float* dV;
  int* globalMin;
  int maxInt = INT_MAX;
  CudaSafeCall(cudaMalloc((void**)&globalMin, sizeof(int)));
  CudaSafeCall(cudaMemcpy(globalMin, &maxInt, sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMalloc((void**)&dV, numPixels*numTimePoints*sizeof(float)));
  CudaSafeCall(cudaMemcpy(dV, V, numPixels*numTimePoints*sizeof(float), cudaMemcpyHostToDevice));
  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  getFlatGridBlock(numPixels*numTimePoints, grid, block);
  ensurePositivity<<<grid,block>>>(dV, numPixels*numTimePoints, globalMin);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(V, dV, numPixels*numTimePoints*sizeof(float), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(dV));
  CudaSafeCall(cudaFree(globalMin));

  std::cout<<"Preparing data for python"<<std::endl;

  npy_intp vdim[] = {numPixels, numTimePoints};
  npy_intp wdim[] = {numPixels, k};
  npy_intp hdim[] = {k, numTimePoints};

  /*
    NOW USE PYTHON TO EXECUTE NNMF WITH TENSORFLOW
  */

  //define python objects
  PyObject *pyV, *pyW, *pyH;
  PyObject *scalarK, *scalarTP, *scalarPix,*scalarIterations;
  PyObject *args;
  PyObject *whReturn;

  //launch python interpreter
  Py_Initialize();
  import_array1();
  if(!Py_IsInitialized()){
    std::cout<<"Error initializing embedded python handler"<<std::endl;
    exit(-1);
  }
  else{
    std::cout<<"Embedded python handler initialized"<<std::endl;
  }

  PyObject* syspath = PySys_GetObject("path");
  PyList_Append(syspath, PyUnicode_FromString("./src"));

  std::cout<<"loading python module"<<std::endl;
  PyObject* myModule = PyImport_ImportModule("tfNNMF");
  if(!myModule){
    std::cout<<"tfNNMF cannot be imported"<<std::endl;
    PyErr_Print();
    exit(-1);
  }
  PyObject* myFunction = PyObject_GetAttrString(myModule, "tensorflowNNMF");

  scalarK = PyLong_FromUnsignedLong(k);
  scalarPix = PyLong_FromUnsignedLong(numPixels);
  scalarTP = PyLong_FromUnsignedLong(numTimePoints);
  scalarIterations = PyLong_FromUnsignedLong(1000);

  std::cout<<"loading V matrix into numpy array"<<std::endl;
  pyV = PyArray_SimpleNew(2, vdim, NPY_FLOAT);
  float* npy = (float *) PyArray_DATA(reinterpret_cast<PyArrayObject*>(pyV));
  for(int i = 0; i < numPixels; ++i){
    memcpy(npy, V + (i*numTimePoints), sizeof(float)*numTimePoints);
    npy += numTimePoints;
  }
  delete[] V;

  args = PyTuple_New(3);
  PyTuple_SetItem(args, 0, pyV);
  PyTuple_SetItem(args, 1, scalarK);
  PyTuple_SetItem(args, 2, scalarIterations);

  whReturn = PyObject_CallObject(myFunction, args);
  if(!whReturn){
    std::cout<<"Error in execution of tfnnmf.py"<<std::endl;
    PyErr_Print();
    exit(-1);
  }

  pyW = PyTuple_GetItem(whReturn, 0);
  pyH = PyTuple_GetItem(whReturn, 1);

  float* tempW;
  float* tempH;

  tempW = (float *) PyArray_GETPTR1(reinterpret_cast<PyArrayObject*>(pyW), 0);
  tempH = (float *) PyArray_GETPTR1(reinterpret_cast<PyArrayObject*>(pyH), 0);
  for(int i = 0; i < numPixels*k; ++i){
    W[i] = tempW[i];
  }
  for(int i = 0;i < k*numTimePoints; ++i){
    H[i] = tempH[i];
  }

  Py_DECREF(syspath);
  Py_DECREF(myFunction);
  Py_DECREF(myModule);
  Py_DECREF(pyV);
  Py_DECREF(pyW);
  Py_DECREF(pyH);
  Py_DECREF(scalarK);
  Py_DECREF(scalarPix);
  Py_DECREF(scalarTP);
  Py_DECREF(scalarIterations);
  Py_Finalize();
}
