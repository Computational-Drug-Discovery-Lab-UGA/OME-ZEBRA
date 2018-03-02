#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "MatrixOperations.cuh"
#include "Utilities.cuh"
#include <ctime>

using namespace std;

#define TILE_DIM 38
#define BLOCK_ROWS 8

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    //err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}

int* flattenMatrix(int** matrix, int cols, int rows){
  int* flat = new int[rows*cols];
  int currentPixel = 0;
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      flat[currentPixel] = matrix[r][c];
      ++currentPixel;
    }
  }
  cout<<"Matrix is flattened."<<endl;
  return flat;
}

int** expandMatrix(int* flattened, int cols, int rows){
  int** expanded = new int*[rows];
  int currentPixel = 0;
  for(int r = 0; r < rows; ++r){
    int* currentRow = new int[cols];
    for(int c = 0; c < cols; ++c){
      currentRow[c] = flattened[currentPixel];
      ++currentPixel;
    }
    expanded[r] = currentRow;
  }
  cout<<"Array is now a Matrix."<<endl;
  return expanded;
}


int** incrementMatrix(int transpose, int** matrix, int cols, int rows){
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      matrix[r][c] += transpose;
    }
  }
  return matrix;
}

__global__ void transposeCoalesced(int *odata, const int *idata){
  __shared__ int tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(int *odata, const int *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];

}

void printDeviceProperties() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf(" -Device name: %s\n", prop.name);
        printf(" -Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf(" -Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf(" -Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf(" -Max number of threads per block: %d\n\n",
               prop.maxThreadsPerBlock);
        printf(" -Max number of blocks: %dx%dx%d\n\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf(" -Total number of multiprocessors: %d\n\n",
               prop.multiProcessorCount);


    }

}
int** hostTranspose(int** matrix, int rows, int cols){
  int** transposable = new int*[rows];
  for(int row = 0; row < rows; ++row){
    transposable[row] = new int[cols];
    for(int col = 0; col < cols; ++col){
      transposable[row][col] = matrix[col][row];
    }
    cout<<"Timepoint "<<row<<" trasposed..."<<endl;

  }

  return transposable;


}

int main(){

  time_t timer = time_t(nullptr);
  printDeviceProperties();

  const int numTimePoints = 25;
  const int rows  = 2048;
  const int columns = 1024;
  int** testMatrix = new int*[rows];
  for(int i = 0; i < rows; ++i){
    testMatrix[i] = new int[columns];
    for(int c = 0; c < columns; ++c){
      testMatrix[i][c] = c;
    }
  }

  cout<<"Done filling test array at "<<timer<<endl;
  int** timePointArray = new int*[numTimePoints];
  for(int i = 0; i < numTimePoints; ++i){

    timePointArray[i] = flattenMatrix(incrementMatrix(1, testMatrix, columns, rows), columns, rows);
  }
  cout<<"Done filling timepoint vector at "<<timer<<endl;

  const int Nrows = rows*columns;
  const int Ncols = numTimePoints;

  //int** transposedMatrix = new int*[Nrows];
  //for(int i = 0; i < Nrows; ++i){
  //  testMatrix[i] = new int[columns];
  //  for(int c = 0; c < Ncols; ++c){
  //    testMatrix[i][c] = 0;
  //  }
  //}
  cout<<"Transpose initiation complete at "<<timer<<endl;

  int** transposedMatrix = hostTranspose(timePointArray, Nrows, Ncols);
  if(transposedMatrix[0] != timePointArray[0] && transposedMatrix[1][0] == timePointArray[0][1]){
    cout<<"SUCCESS IN TRANSPOSITION"<<endl;
  }
  else{
    cout<<"FAILURE IN TRANSPOSITION"<<endl;

    exit(1);
  }
  /*
  cout<<"Printing last timepoint:"<<endl;
  for(int i = 0; i < rows*columns; ++i){
    if(i%columns == 0){
      cout<<endl;
    }
    cout<<timepoints[numTimePoints - 1][i]<<" ";
  }
  cout<<endl;
  */


  //SVD time!!!!
  // --- gesvd only supports Nrows >= Ncols
  // --- column major memory ordering
  //thanks OrangOwlSolutions
  //https://github.com/OrangeOwlSolutions/Linear-Algebra/blob/master/SVD/SVD.cu

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //WE NEED TO TRANSPOSE MATRIX
  //THIS IS BECAUSE WE CURRENTLY HAVE TIMEPOINTS AS ROWS
  //now: matrix = numTimePointsX(rows*columns)
  //needs to be (rows*columns)*numTimePoints
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  // --- cuSOLVE input/output parameters/arrays
  int work_size = 0;
  int *devInfo;
  gpuErrchk(cudaMalloc(&devInfo,sizeof(int)));

  // --- CUDA solver initialization
  cusolverDnHandle_t solver_handle;
  cusolverDnCreate(&solver_handle);

  // --- Setting the host, Nrows x Ncols matrix
  double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
  for(int j = 0; j < Nrows; j++){
    for(int i = 0; i < Ncols; i++){
      h_A[j + i*Nrows] = (i + j*j) * sqrt((double)(i + j));
    }
  }

  // --- Setting the device matrix and moving the host matrix to the device
  double *d_A;
  gpuErrchk(cudaMalloc(&d_A, Nrows * Ncols * sizeof(double)));
  gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

  // --- host side SVD results space
  double *h_U = (double *)malloc(Nrows * Nrows     * sizeof(double));
  double *h_V = (double *)malloc(Ncols * Ncols     * sizeof(double));
  double *h_S = (double *)malloc(min(Nrows, Ncols) * sizeof(double));

  // --- device side SVD workspace and matrices
  double *d_U;
  gpuErrchk(cudaMalloc(&d_U,	Nrows * Nrows     * sizeof(double)));
  double *d_V;
  gpuErrchk(cudaMalloc(&d_V,	Ncols * Ncols	  * sizeof(double)));
  double *d_S;
  gpuErrchk(cudaMalloc(&d_S,	min(Nrows, Ncols) * sizeof(double)));

  // --- CUDA SVD initialization
  cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
  double *work;
  gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));

  // --- CUDA SVD execution
  cusolveSafeCall(cusolverDnDgesvd(solver_handle, 'A', 'A', Nrows, Ncols, d_A, Nrows, d_S, d_U, Nrows, d_V, Ncols, work, work_size, NULL, devInfo));
  int devInfo_h = 0;
  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0){
    std::cout	<< "Unsuccessful SVD execution\n\n";
  }

  // --- Moving the results from device to host
  gpuErrchk(cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_U, d_U, Nrows * Nrows     * sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_V, d_V, Ncols * Ncols     * sizeof(double), cudaMemcpyDeviceToHost));

  std::cout << "Singular values\n";
  for(int i = 0; i < min(Nrows, Ncols); i++){
    std::cout << "d_S["<<i<<"] = " << std::setprecision(15) << h_S[i] << std::endl;
  }
  std::cout << "\nLeft singular vectors - For y = A * x, the columns of U span the space of y\n";
  for(int j = 0; j < Nrows; j++) {
    printf("\n");
    for(int i = 0; i < Nrows; i++)
      printf("U[%i,%i]=%f\n",i,j,h_U[j*Nrows + i]);
  }

  std::cout << "\nRight singular vectors - For y = A * x, the columns of V span the space of x\n";
  for(int i = 0; i < Ncols; i++) {
    printf("\n");
    for(int j = 0; j < Ncols; j++){
      printf("V[%i,%i]=%f\n",i,j,h_V[j*Ncols + i]);
    }
  }

  cusolverDnDestroy(solver_handle);

  /*
  //transposition
  int* flattenedFull = flattenMatrix(timePointArray, Nrows, Ncols);//Nrows and Ncols are switched here
  int* flatTransposed = new int[Nrows*Ncols];
  int* fullDevice;
  int* transDevice;

  if(Nrows % 32 != 0 || Ncols % 32 != 0){
    cout<<"error Nrows or Ncols is not a multiple of 32..."<<endl;
    exit(1);
  }


  dim3 dimGrid(Ncols/TILE_DIM, Nro/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
  CudaSafeCall(cudaMalloc((void**)&fullDevice, Nrows*Ncols*sizeof(int)));
  CudaSafeCall(cudaMalloc((void**)&transDevice, Nrows*Ncols*sizeof(int)));
  CudaSafeCall(cudaMemcpy(fullDevice, flattenedFull, Nrows*Ncols*sizeof(int), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(transDevice, flatTransposed, Nrows*Ncols*sizeof(int), cudaMemcpyHostToDevice));



  CudaSafeCall(cudaMemcpy(flatTransposed, transDevice, Nrows*Ncols*sizeof(int), cudaMemcpyDeviceToHost));
  int** pixelsByTimePoints = expandMatrix(flatTransposed, Ncols, Nrows);


  if(pixelsByTimePoints[0] != timePointArray[0] && pixelsByTimePoints[0][0] == timePointArray[0][1]){
      cout<<"SUCCESS IN TRANSPOSITION"<<endl;
  }
  else{
      cout<<"FAILURE IN TRANSPOSITION"<<endl;
  }
  */
  //
  return 0;

}
