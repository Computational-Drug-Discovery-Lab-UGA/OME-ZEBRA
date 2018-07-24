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

// status printed and convergence check every ITER_CHECK iterations
#define ITER_CHECK 25
// max number of iterations
#define MAX_ITER 200
// set to zero to guarantee MAX_ITER iterations, 0.001 is a good value otherwise
#define CONVERGE_THRESH 0.001

// number of timers used in profiling (don't change)
#define TIMERS 10
char *tname[] = {"total","sgemm","eps","vecdiv","vecmult","sumrows","sumcols","coldiv","rowdiv","check"};



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
    atomicMax(&bmax, value);
    atomicMin(&bmin, value);
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
    normals[globalID] = 1.0f / (1.0f + expf((-10.0f * currentValue) + 7.5));
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
  }
}
__global__ void randInitMatrix(unsigned long size, float* mtx){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  curandState state;
  if(globalID < size){
   curand_init(clock64(), globalID, 0, &state);
   mtx[globalID] = curand_uniform(&state);
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
  CudaSafeCall(cudaMemcpy(matrixDevice, &mtx, size*sizeof(uint32), cudaMemcpyHostToDevice));

  findMinMax<<<grid,block>>>(matrixDevice, size, mind, maxd);
  cudaDeviceSynchronize();
  CudaCheckError();
  normalize<<<grid,block>>>(matrixDevice, normDevice, mind, maxd, size);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(norm, normDevice, size*sizeof(float), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(maxd));
  CudaSafeCall(cudaFree(mind));
  CudaSafeCall(cudaFree(matrixDevice));
  CudaSafeCall(cudaFree(normDevice));

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
  CudaSafeCall(cudaMemcpy(matrixDevice, &mtx, numPixels*numTimePoints*sizeof(float), cudaMemcpyHostToDevice));

  generateKey<<<grid,block>>>(numPixels, numTimePoints, matrixDevice, keyDevice);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(key, keyDevice, numPixels*sizeof(bool), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaFree(matrixDevice));
  CudaSafeCall(cudaFree(keyDevice));
  for(int p = 0; p < numPixels; ++p){
    if(key[p]) ++numPixelsWithValues;
  }

  return key;

}
float* minimizeVideo(unsigned long numPixels, unsigned long numPixelsWithValues, unsigned int numTimePoints, float* mtx, bool* key){
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

double get_time(){
    //output time in microseconds

    //the following line is required for function-wise timing to work,
    //but it slows down overall execution time.
    //comment out for faster execution
    cudaThreadSynchronize();

    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}
int start_time(double* t, int i){
    if (t != NULL)
    {
        t[i] -= get_time();
        return 1;
    }
    else
        return 0;
}
int stop_time(double* t, int i){
    if (t != NULL)
    {
        t[i] += get_time();
        return 1;
    }
    else
        return 0;
}
unsigned nextpow2(unsigned x){
    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x + 1;

}
void update_div(matrix W0, matrix H0, matrix X0, const float thresh, const int max_iter, double *t,int verbose){
    //run iterative multiplicative updates on W,H

    cublasInit();

    const int M = W0.dim[0];
    const int K = W0.dim[1];
    const int N = H0.dim[1];

    // pad matrix dimensions to multiples of:
    const int PAD_MULT = 32;

    int M_padded = M;
    if (M%PAD_MULT != 0)
        M_padded = M + (PAD_MULT - (M % PAD_MULT));

    int K_padded = K;
    if (K%PAD_MULT != 0)
        K_padded = K + (PAD_MULT - (K % PAD_MULT));

    int N_padded = N;
    if (N%PAD_MULT != 0)
        N_padded = N + (PAD_MULT - (N % PAD_MULT));

    //unpadded test
    //M_padded = M;
    //N_padded = N;
    //K_padded = K;

    // find reduction parameters
    int MN_params[4] = {1,1,1,1}; //M*N size reduction (whole matrix)
    int N_params[4] = {1,1,1,1}; //N size reductions (rows)
    int M_params[4] = {1,1,1,1}; //M size reductions (cols)

    int rem;
    rem = nextpow2(N_padded/128 + (!(N_padded%128)?0:1));
    if (rem <= 128)
    {
        N_params[0] = 128;
        N_params[1] = rem;
    }
    else if (rem <= 512)
    {
        N_params[0] = rem;
        N_params[1] = 128;
    }
    else
    {
        fprintf(stderr,"reduction parameter error\n");
        exit(1);
    }


    rem = nextpow2(M_padded/128 + (!(M_padded%128)?0:1));
    if (rem <= 128)
    {
        M_params[0] = 128;
        M_params[1] = rem;
    }
    else if (rem <= 512)
    {
        M_params[0] = rem;
        M_params[1] = 128;
    }
    else
    {
        fprintf(stderr,"reduction parameter error\n");
        exit(1);
    }

    MN_params[0] = M_params[0];
    MN_params[1] = M_params[1];
    MN_params[2] = N_params[0];
    MN_params[3] = N_params[1];

    //printf("reduction parameters: ");
    //printf("%u,%u,%u,%u\n",MN_params[0],MN_params[1],MN_params[2],MN_params[3]);


    // block size in vector arithmetic operations
    const int BLOCK_SIZE = 128;

    //matrix to hold W*H
    matrix WH0;
    create_matrix_on_device(&WH0,M,N,0.0);


    int i;

    /*
       double t_array[TIMERS];
       if(t==NULL)
       t = t_array;
       */
    if (t != NULL)
    {
        for(i=0;i<TIMERS;i++)
            t[i] = 0;
    }

    //float nancheck, zerocheck;
    // compute initial divergence and error
    float diff,div,change,prev_diff,prev_div;
    matrix_multiply_d(W0,H0,WH0);
    diff = matrix_difference_norm_d(compute,X0,WH0,MN_params);


    div = matrix_div_d(compute,X0,WH0,MN_params);
    if(verbose)
        printf("i: %4i, error: %6.4f, initial div: %8.4e\n",0,diff,div);


    // free device memory for unpadded matrices
    free_matrix_on_device(&W0);
    free_matrix_on_device(&H0);
    free_matrix_on_device(&X0);
    free_matrix_on_device(&WH0);


    //initialize temp matrices -----------------------


    //matrix to hold X./(W*H+EPS)
    matrix Z;
    create_matrix_on_device(&Z,M_padded,N_padded,0.0);

    //matrix to hold W'*Z
    matrix WtZ;
    create_matrix_on_device(&WtZ,K_padded,N_padded,0.0);

    //matrix to hold Z*H'
    matrix ZHt;
    create_matrix_on_device(&ZHt,M_padded,K_padded,0.0);

    //matrix to hold sum(W) [sum of cols of W]
    matrix sumW;
    create_matrix_on_device(&sumW,1,K_padded,0.0);

    //matrix to hold sum(H,2) [sum of rows of H]
    matrix sumH2;
    create_matrix_on_device(&sumH2,K_padded,1,0.0);


    //matrices to hold padded versions of matrices
    matrix W;
    create_matrix_on_device(&W,M_padded,K_padded,0.0);

    matrix H;
    create_matrix_on_device(&H,K_padded,N_padded,0.0);

    matrix X;
    create_matrix_on_device(&X,M_padded,N_padded,0.0);




    // move host matrices to padded device memory
    copy_matrix_to_device_padded(W0,W);
    copy_matrix_to_device_padded(H0,H);
    copy_matrix_to_device_padded(X0,X);




    //t[0] -= get_time();
    start_time(t,0);

        //matrix test1;

        for(i=0;i<max_iter;i++){

            //check for convergence, print status
            if(i % ITER_CHECK == 0 && i != 0){
                //t[9] -= get_time();
                start_time(t,9);
                matrix_multiply_d(W,H,Z);
                prev_diff = diff;
                diff = matrix_difference_norm_d(compute,X,Z,MN_params);
                change = (prev_diff-diff)/prev_diff;
                //t[9] += get_time();
                stop_time(t,9);
                if(verbose)
                    printf("i: %4i, error: %6.4f, %% change: %8.5f\n",
                            i,diff,change);
                if(change < thresh){
                    printf("converged\n");
                    break;
                }
            }


            /* matlab algorithm
               Z = X./(W*H+eps); H = H.*(W'*Z)./(repmat(sum(W)',1,F));
               Z = X./(W*H+eps);
               W = W.*(Z*H')./(repmat(sum(H,2)',N,1));
               */

            //
            // UPDATE H -----------------------------
            //


            //WH = W*H
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_d(W,H,Z);
            //t[1] += get_time();
            stop_time(t,1);




            //WH = WH+EPS
            //t[2] -= get_time();
            start_time(t,2);
            matrix_eps_d(Z,BLOCK_SIZE);
            //t[2] += get_time();
            stop_time(t,2);


            //Z = X./WH
            //t[3] -= get_time();
            start_time(t,3);
            element_divide_d(X,Z,Z,BLOCK_SIZE);
            //t[3] += get_time();
            stop_time(t,3);


            //sum cols of W into row vector
            //t[6] -= get_time();
            start_time(t,6);
            sum_cols_d(compute,W,sumW,M_params);
            matrix_eps_d(sumW,32);
            //t[6] += get_time();
            stop_time(t,6);

            //convert sumW to col vector (transpose)
            sumW.dim[0] = sumW.dim[1];
            sumW.dim[1] = 1;


            //WtZ = W'*Z
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_AtB_d(W,Z,WtZ);
            //t[1] += get_time();
            stop_time(t,1);


            //WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
            //[element divide cols of WtZ by sumW']
            //t[7] -= get_time();
            start_time(t,7);
            col_divide_d(WtZ,sumW,WtZ);
            //t[7] += get_time();
            stop_time(t,7);



            //H = H.*WtZ
            //t[4] -= get_time();
            start_time(t,4);
            element_multiply_d(H,WtZ,H,BLOCK_SIZE);
            //t[4] += get_time();
            stop_time(t,4);



            //
            // UPDATE W ---------------------------
            //

            //WH = W*H
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_d(W,H,Z);
            //t[1] += get_time();
            stop_time(t,1);


            //WH = WH+EPS
            //t[2] -= get_time();
            start_time(t,2);
            matrix_eps_d(Z,BLOCK_SIZE);
            //t[2] += get_time();
            stop_time(t,2);

            //Z = X./WH
            //t[3] -= get_time();
            start_time(t,3);
            element_divide_d(X,Z,Z,BLOCK_SIZE);
            //t[3] += get_time();
            stop_time(t,3);


            //sum rows of H into col vector
            //t[5] -= get_time();
            start_time(t,5);
            sum_rows_d(compute,H,sumH2,N_params);
            matrix_eps_d(sumH2,32);
            //t[5] += get_time();
            stop_time(t,5);

            //convert sumH2 to row vector (transpose)
            sumH2.dim[1] = sumH2.dim[0];
            sumH2.dim[0] = 1;

            //ZHt = Z*H'
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_ABt_d(Z,H,ZHt);
            //t[1] += get_time();
            stop_time(t,1);

            //ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
            //[element divide rows of ZHt by sumH2']
            //t[8] -= get_time();
            start_time(t,8);
            row_divide_d(ZHt,sumH2,ZHt);
            //t[8] += get_time();
            stop_time(t,8);

            //W = W.*ZHt
            //t[4] -= get_time();
            start_time(t,4);
            element_multiply_d(W,ZHt,W,BLOCK_SIZE);
            //t[4] += get_time();
            stop_time(t,4);


            // ------------------------------------

            //reset sumW to row vector
            sumW.dim[1] = sumW.dim[0];
            sumW.dim[0] = 1;
            //reset sumH2 to col vector
            sumH2.dim[0] = sumH2.dim[1];
            sumH2.dim[1] = 1;

            // ---------------------------------------

        }

    //t[0] += get_time();
    stop_time(t,0);




    //reallocate unpadded device memory
    allocate_matrix_on_device(&W0);
    allocate_matrix_on_device(&H0);

    //copy padded matrix to unpadded matrices
    copy_from_padded(W0,W);
    copy_from_padded(H0,H);

    // free padded matrices
    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);

    // free temp matrices
    destroy_matrix(&Z);
    destroy_matrix(&WtZ);
    destroy_matrix(&ZHt);
    destroy_matrix(&sumW);
    destroy_matrix(&sumH2);

    copy_matrix_to_device(&X0);
    create_matrix_on_device(&WH0,M,N,0.0);

    // copy device results to host memory
    copy_matrix_from_device(&W0);
    copy_matrix_from_device(&H0);

    // evaluate final results
    matrix_multiply_d(W0,H0,WH0);
    prev_diff = diff;
    diff = matrix_difference_norm_d(compute,X0,WH0,MN_params);
    prev_div = div;
    div = matrix_div_d(compute,X0,WH0,MN_params);
    if(verbose){
        change = (prev_diff-diff)/prev_diff;
        printf("max iterations reached\n");
        printf("i: %4i, error: %6.4f, %% change: %8.5f\n",
                i,diff,change);
        change = (prev_div-div)/prev_div;
        printf("\tfinal div: %8.4e, %% div change: %8.5f\n",
                div,change);

        printf("\n");
        if (t != NULL)
        {
            for(i=0;i<TIMERS;i++)
                printf("t[%i]: %8.3f (%6.2f %%) %s\n",i,t[i],t[i]/t[0]*100,tname[i]);
        }
    }

    //clean up extra reduction memory
    matrix_difference_norm_d(cleanup,X0,WH0,MN_params);
    matrix_div_d(cleanup,X0,WH0,MN_params);
    sum_cols_d(cleanup,W,sumW,M_params);
    sum_rows_d(cleanup,H,sumH2,N_params);

    // free device memory for unpadded matrices
    free_matrix_on_device(&W0);
    free_matrix_on_device(&H0);
    free_matrix_on_device(&X0);

    // free temp matrices
    destroy_matrix(&WH0);

    cublasShutdown();

}

void performNNMF(float* &W, float* &H, float* V, unsigned int k, unsigned long numPixels, unsigned int numTimePoints){
  float* dW;
  float* dH;
  float* dMatrix;
  CudaSafeCall(cudaMalloc((void**)&dW, numPixels*k*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&dH, k*numTimePoints*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&dMatrix, numPixels*numTimePoints*sizeof(float)));
  CudaSafeCall(cudaMemcpy(dMatrix, V, numPixels*numTimePoints*sizeof(float), cudaMemcpyHostToDevice));
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

  matrix wM, hM, vM;
  wM.dim[0] = numPixels;
  wM.dim[1] = k;
  hM.dim[0] = k;
  hM.dim[1] = numTimePoints;
  vM.dim[0] = numPixels;
  vM.dim[1] = numTimePoints;
  wM.mat = W;
  hM.mat = H;
  vM.mat = V;
  wM.mat_d = dW;
  hM.mat_d = dH;
  vM.mat_d = dMatrix;
  matrix_eps(wM);
  matrix_eps(hM);
  matrix_eps(vM);
  matrix_eps_d(wM, 32);
  matrix_eps_d(hM, 32);
  matrix_eps_d(vM, 32);

  update_div(wM, hM, vM, CONVERGE_THRESH, MAX_ITER, NULL, 1);

  CudaSafeCall(cudaFree(wM.mat_d));
  CudaSafeCall(cudaFree(hM.mat_d));
  CudaSafeCall(cudaFree(vM.mat_d));
}
