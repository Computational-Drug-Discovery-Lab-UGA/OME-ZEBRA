#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "MatrixOperations.cuh"
#include <ctime>
#include<iostream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

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

float* flattenMatrix(float** matrix, int cols, int rows){
  float* flat = new float[rows*cols];
  int currentPixel = 0;
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      flat[currentPixel] = matrix[r][c];
      ++currentPixel;
    }
  }
  //cout<<"Matrix is flattened."<<endl;
  return flat;
}

float** expandMatrix(float* flattened, int cols, int rows){
  float** expanded = new float*[rows];
  int currentPixel = 0;
  for(int r = 0; r < rows; ++r){
    float* currentRow = new float[cols];
    for(int c = 0; c < cols; ++c){
      currentRow[c] = flattened[currentPixel];
      ++currentPixel;
    }
    expanded[r] = currentRow;
  }
  cout<<"Array is now a Matrix."<<endl;
  return expanded;
}


float** incrementMatrix(float alter, float** matrix, int cols, int rows){
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      matrix[r][c] += alter;
    }
  }
  return matrix;
}


float** hostTranspose(float** matrix, int rows, int cols){
  float** transposable = new float*[rows];
  for(int row = 0; row < rows; ++row){
    transposable[row] = new float[cols];
    for(int col = 0; col < cols; ++col){
      transposable[row][col] = matrix[col][row];
    }
    //cout<<"Timepoint "<<row<<" trasposed..."<<endl;

  }

  return transposable;
}

__global__ void transposefloatMatrix(float* flatOrigin, float* flatTransposed, long Nrows, long Ncols){

  long globalID = blockIdx.x * blockDim.x + threadIdx.x;
  long pixel = globalID;
  long stride = gridDim.x * blockDim.x;
  long flatLength = Nrows * Ncols;
  long row = 0;
  long col = 0;
  float currentPixelIntensity = 0;
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
  float** testMatrix = new float*[rows];
  for(int i = 0; i < rows; ++i){
    testMatrix[i] = new float[columns];
    for(int c = 0; c < columns; ++c){
      testMatrix[i][c] = c;
    }
  }

  cout<<"Done filling test array at "<<difftime(time(nullptr), timer)<<" second"<<endl;
  float** timePointArray = new float*[numTimePoints];
  for(int i = 0; i < numTimePoints; ++i){

    timePointArray[i] = flattenMatrix(incrementMatrix(1, testMatrix, columns, rows), columns, rows);
  }
  cout<<"Done filling timepoint vector at "<<difftime(time(nullptr), timer)<<" second"<<endl;



  bool transposed = false;
  int Nrows = 0;
  int Ncols = 0;
  float* flattenedFull = flattenMatrix(timePointArray, rows*columns, numTimePoints);//Nrows and Ncols are switched here
  cout<<"Original Array has been flattened"<<endl;
  float* flatTransposed = new float[rows*columns*numTimePoints];//might not be used
  float* fullDevice;
  float* transDevice;
  float* deviceSVDMatrix;
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

    float** transposedMatrix = hostTranspose(timePointArray, Nrows, Ncols);
    if(transposedMatrix[0] != timePointArray[0] && transposedMatrix[1][0] == timePointArray[0][1]){
      cout<<"SUCCESS IN TRANSPOSITION IN>>>"<<difftime(time(nullptr), transposeTimer)<<" second"<<endl;
    }
    else{
      cout<<"FAILURE IN TRANSPOSITION"<<endl;

      exit(1);
    }


    //transposition



    CudaSafeCall(cudaMalloc((void**)&fullDevice, Nrows*Ncols*sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&transDevice, Nrows*Ncols*sizeof(float)));
    CudaSafeCall(cudaMemcpy(fullDevice, flattenedFull, Nrows*Ncols*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(transDevice, flatTransposed, Nrows*Ncols*sizeof(float), cudaMemcpyHostToDevice));


    unsigned int numBlocks = 2147483647;
    while(Nrows * Ncols < numBlocks){
      numBlocks--;
    }
    long startIndex = 0;
    long endIndex = numBlocks;
    transposeTimer = time(nullptr);
    cout<<"LENGTH OF FLAT MATRIX = "<<Nrows * Ncols<<endl;
    cout<<"MATRIX DIM = "<<Nrows <<"x"<<Ncols<<endl;
    transposefloatMatrix<<<numBlocks,1>>>(fullDevice, transDevice, Nrows, Ncols);
    CudaCheckError();

    CudaSafeCall(cudaMemcpy(flatTransposed, transDevice, Nrows*Ncols*sizeof(float), cudaMemcpyDeviceToHost));
    float** pixelsByTimePoints = expandMatrix(flatTransposed, Ncols, Nrows);

    CudaSafeCall(cudaFree(fullDevice));
    CudaSafeCall(cudaFree(transDevice));

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

  float* flatSVDMatrix;

  if(transposed){//use flatTransposed
    flatSVDMatrix = flatTransposed;
    CudaSafeCall(cudaMalloc((void**)&deviceSVDMatrix, Nrows*Ncols*sizeof(float)));
    CudaSafeCall(cudaMemcpy(deviceSVDMatrix, flatTransposed, Nrows*Ncols*sizeof(float), cudaMemcpyHostToDevice));
  }
  else{//use flattenedFull
    flatSVDMatrix = flattenedFull;
    CudaSafeCall(cudaMalloc((void**)&deviceSVDMatrix, Nrows*Ncols*sizeof(float)));
    CudaSafeCall(cudaMemcpy(deviceSVDMatrix, flattenedFull, Nrows*Ncols*sizeof(float), cudaMemcpyHostToDevice));
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
  // --- cuSOLVE input/output parameters/arrays


  //may have to be done by matlab



  return 0;

}
