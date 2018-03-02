#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "MatrixOperations.cuh"
#include "Utilities.cuh"
#include <ctime>

using namespace std;


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

short* flattenMatrix(short** matrix, int cols, int rows){
  short* flat = new short[rows*cols];
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

short** expandMatrix(short* flattened, int cols, int rows){
  short** expanded = new short*[rows];
  int currentPixel = 0;
  for(int r = 0; r < rows; ++r){
    short* currentRow = new short[cols];
    for(int c = 0; c < cols; ++c){
      currentRow[c] = flattened[currentPixel];
      ++currentPixel;
    }
    expanded[r] = currentRow;
  }
  cout<<"Array is now a Matrix."<<endl;
  return expanded;
}


short** incrementMatrix(short alter, short** matrix, int cols, int rows){
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      matrix[r][c] += alter;
    }
  }
  return matrix;
}


short** hostTranspose(short** matrix, int rows, int cols){
  short** transposable = new short*[rows];
  for(int row = 0; row < rows; ++row){
    transposable[row] = new short[cols];
    for(int col = 0; col < cols; ++col){
      transposable[row][col] = matrix[col][row];
    }
    //cout<<"Timepoint "<<row<<" trasposed..."<<endl;

  }

  return transposable;
}

__global__ void transposeShortMatrix(short* flatOrigin, short* flatTransposed, long Nrows, long Ncols){

  long globalID = blockIdx.x * blockDim.x + threadIdx.x;
  long pixel = globalID;
  long stride = gridDim.x * blockDim.x;
  long flatLength = Nrows * Ncols;
  long row = 0;
  long col = 0;
  short currentPixelIntensity = 0;
  while(pixel < flatLength){
    row = pixel/Ncols;
    col = pixel - Ncols*row;
    flatTransposed[pixel] = flatOrigin[row + Nrows*col];
    pixel += stride;
  }

}

int main(){

  time_t timer = time(nullptr);
  printDeviceProperties();

  int numTimePoints = 512;
  int rows  = 2048;
  const int columns = 1024;
  short** testMatrix = new short*[rows];
  for(int i = 0; i < rows; ++i){
    testMatrix[i] = new short[columns];
    for(int c = 0; c < columns; ++c){
      testMatrix[i][c] = c;
    }
  }

  cout<<"Done filling test array at "<<difftime(time(nullptr), timer)<<" second"<<endl;
  short** timePointArray = new short*[numTimePoints];
  for(int i = 0; i < numTimePoints; ++i){

    timePointArray[i] = flattenMatrix(incrementMatrix(1, testMatrix, columns, rows), columns, rows);
  }
  cout<<"Done filling timepoint vector at "<<difftime(time(nullptr), timer)<<" second"<<endl;



  bool transposed = false;
  int Nrows = 0;
  int Ncols = 0;
  short* flattenedFull = flattenMatrix(timePointArray, rows*columns, numTimePoints);//Nrows and Ncols are switched here
  cout<<"Original Array has been flattened"<<endl;
  short* flatTransposed = new short[rows*columns*numTimePoints];//might not be used
  if(rows*columns >= numTimePoints){
    transposed = true;
    Nrows = rows*columns;
    Ncols = numTimePoints;
    //int** transposedMatrix = new int*[Nrows];
    //for(int i = 0; i < Nrows; ++i){
    //  testMatrix[i] = new int[columns];
    //  for(int c = 0; c < Ncols; ++c){
    //    testMatrix[i][c] = 0;
    //  }
    //}
    cout<<"Transpose initiation complete at "<<difftime(time(nullptr), timer)<<" second"<<endl;

    time_t transposeTimer = time(nullptr);

    short** transposedMatrix = hostTranspose(timePointArray, Nrows, Ncols);
    if(transposedMatrix[0] != timePointArray[0] && transposedMatrix[1][0] == timePointArray[0][1]){
      cout<<"SUCCESS IN TRANSPOSITION IN>>>"<<difftime(time(nullptr), transposeTimer)<<" second"<<endl;
    }
    else{
      cout<<"FAILURE IN TRANSPOSITION"<<endl;

      exit(1);
    }


    //transposition
    short* fullDevice;
    short* transDevice;


    CudaSafeCall(cudaMalloc((void**)&fullDevice, Nrows*Ncols*sizeof(short)));
    CudaSafeCall(cudaMalloc((void**)&transDevice, Nrows*Ncols*sizeof(short)));
    CudaSafeCall(cudaMemcpy(fullDevice, flattenedFull, Nrows*Ncols*sizeof(short), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(transDevice, flatTransposed, Nrows*Ncols*sizeof(short), cudaMemcpyHostToDevice));


    unsigned int numBlocks = 2147483647;
    while(Nrows * Ncols < numBlocks){
      numBlocks--;
    }
    long startIndex = 0;
    long endIndex = numBlocks;
    transposeTimer = time(nullptr);
    cout<<"LENGTH OF FLAT MATRIX = "<<Nrows * Ncols<<endl;
    cout<<"MATRIX DIM = "<<Nrows <<"x"<<Ncols<<endl;
    transposeShortMatrix<<<numBlocks,1>>>(fullDevice, transDevice, Nrows, Ncols);
    CudaCheckError();

    CudaSafeCall(cudaMemcpy(flatTransposed, transDevice, Nrows*Ncols*sizeof(short), cudaMemcpyDeviceToHost));
    short** pixelsByTimePoints = expandMatrix(flatTransposed, Ncols, Nrows);


    if(pixelsByTimePoints[0] != timePointArray[0] && pixelsByTimePoints[1][0] == timePointArray[0][1]){
        cout<<"SUCCESS IN TRANSPOSITION KERNEL IN>>>"<<difftime(time(nullptr), transposeTimer)<<" second"<<endl;
    }
    else{
        cout<<"FAILURE IN TRANSPOSITION KERNEL"<<endl;
        exit(1);

    }
  }
  else{
    Nrows = numTimePoints;
    Ncols = rows*columns;
  }

  short* flatSVDMatrix;

  if(transposed){//use flatTransposed
    flatSVDMatrix = flatTransposed;
  }
  else{//use flattenedFull
    flatSVDMatrix = flattenedFull;
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

/*
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
*/


  return 0;

}
