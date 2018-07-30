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

void executeMultiplyMatrices(float *matrixA, float *matrixB, float* &matrixC,
                                 long diffDimA, long comDim, long diffDimB){

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


void performSVD(unsigned int numSigFig, unsigned long mValue, unsigned long nValue, float* originalMatrix, float* &sMatrix, float* &uMatrix, float* &vtMatrix) {
  std::cout<<"starting SVD"<<std::endl;
  magma_init(); // initialize Magma
  real_Double_t gpu_time, cpu_time;

  magma_int_t m=mValue, n=nValue, n2=m*n, min_mn = (((m)<(n))?(m):(n));
  printf("m=%d,n=%d\n",m,n);
  float *r; // a,r - mxn matrices
  float *u, *vt;// u - mxm matrix , vt - nxn matrix on the host
  float *s1; // vectors of singular values
  magma_int_t info;
  magma_int_t ione = 1;
  float work[1], error = 1.; // used in difference computations
  float mone = -1.0, * h_work ; // h_work - workspace
  float dummy[1];
  magma_int_t lwork ; // workspace size
  magma_int_t ISEED[4] = {0 ,0 ,0 ,1}; // seed

  // Allocate host memory
  magma_smalloc_cpu(&vt, n*n); // host memory for vt
  magma_smalloc_cpu(&u, m*n); // host memory for u
  magma_smalloc_cpu(&s1 , min_mn ); // host memory for s1
  magma_smalloc_pinned(&r,n2 ); // host memory for r
  magma_int_t nb = magma_get_sgesvd_nb(m,n); // optim . block size

  magma_sgesvd(MagmaSomeVec, MagmaSomeVec, m, n,
                NULL, m, NULL, NULL, m, NULL, n, dummy, -1,
                &info );

  lwork = (magma_int_t) MAGMA_S_REAL( dummy[0] );
  magma_smalloc_pinned(& h_work , lwork ); // host mem . for h_work



  // lapackf77_slarnv(&ione, ISEED, &n2, a);
  lapackf77_slacpy(MagmaFullStr, &m, &n, originalMatrix, &m, r, &m);

  // MAGMA
  gpu_time = magma_wtime();

  // compute the singular value decomposition of r ( copy of a)
  // and optionally the left and right singular vectors :
  // r = u* sigma *vt; the diagonal elements of sigma (s1 array )
  // are the singular values of a in descending order
  // the first min (m,n) columns of u contain the left sing . vec .
  // the first min (m,n) columns of vt contain the right sing .vec .
  magma_sgesvd(MagmaSomeVec,MagmaSomeVec,m,n,r,m,s1,u,m,vt,n,h_work,
  lwork,&info );

  gpu_time = magma_wtime() - gpu_time ;
  printf(" sgesvd gpu time: %7.5f\n", gpu_time); // Magma time
  std::cout<<"starting minimization with num singular values being "<<numSigFig<<std::endl;
  for (long i = 0; i < numSigFig; i++) {
    for (long j = 0; j < numSigFig; j++) {
      if (j == i) {
        sMatrix[i * numSigFig + j] = s1[i];
      }
      else {
        sMatrix[i * numSigFig + j] = 0.0f;
      }
    }
  }

  std::cout<<"sMatrix minimized = "<<numSigFig<<"x"<<numSigFig<<std::endl;

  for(long i = 0; i < m; i++){
    for(int j = 0; j < numSigFig; j++) {
        uMatrix[i*n + j] = u[i*n + j];
    }
  }

  std::cout<<"uMatrix minimized = "<<m<<"x"<<numSigFig<<std::endl;

  for(long i = 0; i < numSigFig; i++) {

    for(int j = 0; j < n; j++) {

        vtMatrix[i*n + j] = vt[i*n + j];

    }

  }

  std::cout<<"vtMatrix minimized = "<<numSigFig<<"x"<<n<<std::endl;


  // values
  // Free memory
  free(vt); // free host memory
  free(s1); //free host memory
  free(u); // free host memory
  magma_free_pinned( h_work ); // free host memory
  magma_free_pinned(r); // free host memory
  magma_finalize( ); // finalize Magma

}

void performNNMF(float* &W, float* &H, float* V, unsigned int k, unsigned long numPixels, unsigned int numTimePoints, std::string baseDir, bool svd){
  float* svdProduct;
  unsigned int numSigFig;
  float* sMatrix;
  float* uMatrix;
  float* vtMatrix;
  float* tempMatrix;
  if(svd){
    numSigFig = 200;
    /*DO SVD*/
    sMatrix = new float[numSigFig*numSigFig];
    uMatrix = new float[numPixels*numSigFig];
    vtMatrix = new float[numSigFig*numTimePoints];
    performSVD(numSigFig, numPixels, numTimePoints, V, sMatrix, uMatrix, vtMatrix);
    delete[] V;
    tempMatrix = new float[numPixels*numSigFig];
    executeMultiplyMatrices(uMatrix, sMatrix, tempMatrix, numPixels, numSigFig, numSigFig);
    delete[] uMatrix;
    delete[] sMatrix;
    svdProduct = new float[numPixels*numTimePoints];
    executeMultiplyMatrices(tempMatrix, vtMatrix, svdProduct, numPixels, numSigFig, numTimePoints);
    delete[] vtMatrix;
    delete[] tempMatrix;
    numSigFig = 50;
    sMatrix = new float[numSigFig*numSigFig];
    uMatrix = new float[numPixels*numSigFig];
    vtMatrix = new float[numSigFig*numTimePoints];
    performSVD(numSigFig, numPixels, numTimePoints, svdProduct, sMatrix, uMatrix, vtMatrix);
    delete[] svdProduct;
    tempMatrix = new float[numPixels*numSigFig];
    executeMultiplyMatrices(uMatrix, sMatrix, tempMatrix, numPixels, numSigFig, numSigFig);
    delete[] uMatrix;
    delete[] sMatrix;
    svdProduct = new float[numPixels*numTimePoints];
    executeMultiplyMatrices(tempMatrix, vtMatrix, svdProduct, numPixels, numSigFig, numTimePoints);
    delete[] vtMatrix;
    delete[] tempMatrix;
  }
  else{
    svdProduct = V;
  }

  clock_t nnmfTimer;
  nnmfTimer = clock();
  std::cout<<"starting nnmf"<<std::endl;
  float min = std::numeric_limits<float>::max();
  for(int i = 0; i < numPixels*numTimePoints; ++i){
    if(svdProduct[i] < min) min = svdProduct[i];
  }
  for(int i = 0; i < numPixels*numTimePoints; ++i){
    svdProduct[i] -= (min - .1);
  }
  /*WRITE NNMF.txt */
  std::string nmfFileName = baseDir + "NNMF.txt";
  std::ofstream NNMFile(nmfFileName);
  if(NNMFile.is_open()){
    for(int i = 0; i < numPixels*numTimePoints; ++i){
      if ((i + 1) % numTimePoints == 0) {
        NNMFile << svdProduct[i] << "\n";
      }
      else {
        NNMFile << svdProduct[i] << " ";
      }
    }
    NNMFile.close();
    std::cout<< nmfFileName <<" has been created.\n"<<std::endl;
  }
  else{
    std::cout<<"error cannot create"<< nmfFileName <<std::endl;
  }
  printf("writing NNMF.txt took %f seconds.\n\n", ((float) clock() - nnmfTimer)/CLOCKS_PER_SEC);
  nnmfTimer = clock();
  delete[] svdProduct;

  /*DO NMF*/
  std::string executableLine = "./bin/NMF_GPU " + baseDir + "NNMF.txt -k " + std::to_string(k) + " -j 10 -t 40 -i 20000";
  std::system(executableLine.c_str());


  printf("nnmf took %f seconds.\n\n", ((float) clock() - nnmfTimer)/CLOCKS_PER_SEC);
  nnmfTimer = clock();
  W = new float[k*numPixels];
  H = new float[k*numTimePoints];
  std::cout<<"reading in h and w file"<<std::endl;
  std::string wFileName = nmfFileName + "_W.txt";
  std::string hFileName = nmfFileName + "_H.txt";
  std::cout<<"opening "<<wFileName<<" and"<<hFileName<<std::endl;
  std::string wLine = "";
  std::string hLine = "";
  std::ifstream wFile(wFileName);
  std::ifstream hFile(hFileName);
  std::istringstream hh;
  std::istringstream ww;
  if(wFile.is_open() && hFile.is_open()){
    for(int row = 0; row < numPixels; ++row){
      wLine = "";
      hLine = "";
      if(row < k){
        getline(hFile, hLine);
        hh = std::istringstream(hLine);
      }
      getline(wFile, wLine);
      ww = std::istringstream(wLine);
      for(int col = 0; col < k; ++col){
        ww >> W[row*k + col];
      }
      for(int col = 0; row < k && col < numTimePoints; ++col){
        hh >> H[row*numTimePoints + col];
      }
    }
    wFile.close();
    hFile.close();
  }
  else{
    std::cout<<"error cannot open W or H file"<<std::endl;
  }
  printf("reading h and w took %f seconds.\n\n", ((float) clock() - nnmfTimer)/CLOCKS_PER_SEC);

}
