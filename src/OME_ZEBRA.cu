#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include "tiffio.h"
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

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





void printDeviceProperties();
void printArray(uint32 * array, uint32 width);
uint32* extractMartrices(TIFF* tif, string fileName);
uint32* extractMartrices(TIFF* tif, string fileName, int currentTimePoint);
uint32* extractMartrices(TIFF* tif);
vector<uint32> flattenMatrix(vector<uint32*> matrix, int cols, int rows);
uint32** hostTranspose(uint32** matrix, int rows, int cols);
__global__ void transposeuint32Matrix(uint32* flatOrigin, uint32* flatTransposed, long Nrows, long Ncols);
uint32 findMin(uint32* flatMatrix, int size);
__global__ void calcCa(uint32* flatMatrix, uint32 min, long size);
__global__ void fillTestMatrix(uint32* flatMatrix, long size);
double calculateStandardDeviation(double * subVarX, double lengthOfSet);
double calculateCoVariance(double * subVarX, double * subVarY, double lengthOfSets);
void calculateSubvariance(uint32 * inputSet, double lengthOfSet, double * resultArray);
double calculateAverage(uint32 * inputSet, double lengthOfSet);
double calculatePearsonCorrelationCoefficient(uint32 * x, uint32 * y, double lengthOfSets);
void transposeArray(vector<uint32*> inputArray, int n, int m, uint32 * outputArray, uint32 & min, uint32 & max);




int main(int argc, char *argv[]) {

    if(argc != 2) {
      cout << "Usage: ./exe <file>";
      return 1;
    }
    else {
      vector<uint32*> flattenedTimePoints;
      TIFF* tif = TIFFOpen(argv[1], "r");
      cout<<endl<<argv[1]<<" IS OPENED\n"<<endl;
      string fileName = argv[1];
      int dircount = 0;
      uint32 numColumns;
      uint32 numRows;
      if (tif) {
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &numColumns);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &numRows);
          do {
            uint32* flatMatrix = new uint32[numRows*numColumns];
            if(dircount == 100){
              flatMatrix = extractMartrices(tif, fileName, dircount);
              //exit(0);
            }
            if(dircount == 0){
              flatMatrix = extractMartrices(tif, fileName);
            }

            else flatMatrix = extractMartrices(tif);

            flattenedTimePoints.push_back(flatMatrix);
            dircount++;
          }
          while (TIFFReadDirectory(tif));
          printf("%d directories in %s\n", dircount, argv[1]);

          //to save numColumns


          TIFFClose(tif);


          //NO NEED TO DO THIS
          //prepare arrays
          //flatten time points
          //cout << "Preparing to flatten" << endl;
          //vector<uint32> fullTiffVector = flattenMatrix(flattenedTimePoints, numRows*numColumns, dircount);

          cout << "Flattening Array" << endl;

          int NNormal = dircount;
          int MNormal = (numRows*numColumns);
          int totalSize = MNormal*NNormal;

          uint32 min = 4294967295;
          uint32 max = 0;
          uint32* actualArray = new uint32[totalSize];

          transposeArray(flattenedTimePoints, NNormal, MNormal, actualArray, min, max);

          cout << "loading arrays based on key" << endl;

          dim3 grid = {1,1,1};
          dim3 block = {1,1,1};
          if(65535 > totalSize){
            grid.x = totalSize;
          }
          else if(65535*1024 > totalSize){
            grid.x = 65535;
            block.x = 1024;
            while(block.x*grid.x > totalSize){
              block.x--;
            }
          }
          else{
            grid.x = 65535;
            block.x = 1024;
            while(grid.x*grid.y*block.x < totalSize){
              grid.y++;
            }
          }
          cout<<"prepare for calcCa cuda kernel with min = "<<min<<",max = "<<max<<endl;
          uint32* actualArrayDevice;
          CudaSafeCall(cudaMalloc((void**)&actualArrayDevice,totalSize*sizeof(uint32)));
          CudaSafeCall(cudaMemcpy(actualArrayDevice,actualArray, totalSize*sizeof(uint32), cudaMemcpyHostToDevice));
          calcCa<<<grid,block>>>(actualArrayDevice, min, totalSize);
          CudaCheckError();
          CudaSafeCall(cudaMemcpy(actualArray,actualArrayDevice, totalSize*sizeof(uint32), cudaMemcpyDeviceToHost));
          CudaSafeCall(cudaFree(actualArrayDevice));
          cout << "calcCa has completed applying offset" << endl;

          // cout << "Starting Pearson Correlation Coefficient processing" << endl;
          //
          // uint32 * firstPoint = new uint32[512];
          // uint32 * secondPoint = new uint32[512];
          //
          // long dimOfArray1 = 512;
          // long dimOfArray2 = 1024;
          //
          // unsigned long sizeOfPearson = (((dimOfArray1 * dimOfArray2) * (dimOfArray1 * dimOfArray2)) - (dimOfArray1 * dimOfArray2))/2;
          //
          // cout << "pearsonArray created of size: " << sizeOfPearson << endl;
          //
          // double * pearsonArray = new double[sizeOfPearson];
          // sizeOfPearson = 0;
          //
          // for (int i = 0; i < (512*1024); i++) {
          //
          //   for (int index1 = 0; index1 < 512; index1++) {
          //
          //     firstPoint[index1] = i*512 + index1;
          //
          //   }
          //
          //   for (int j = (i+1); j < (512*1024); j++) {
          //
          //     for (int index2 = 0; index2 < 512; index2++) {
          //
          //       secondPoint[index2] = j*512 + index2;
          //
          //     }
          //
          //     pearsonArray[sizeOfPearson] = calculatePearsonCorrelationCoefficient(firstPoint, secondPoint, 512);
          //     sizeOfPearson++;
          //
          //   }
          //
          //   if (i % 1000 == 0) {
          //
          //     cout << i << endl;
          //
          //   }
          //
          // }
          //
          // ofstream myPearsonFile ("data/pearsonArray.txt");
          // if (myPearsonFile.is_open()) {
          //
          //   for(int i = 0; i < sizeOfPearson; i++){
          //
          //     myPearsonFile << pearsonArray[i];
          //
          //     if (i != (sizeOfPearson - 1)) {
          //
          //       myPearsonFile << "\n";
          //
          //     }
          //
          //   }
          //
          //   myPearsonFile.close();
          //
          // }

          cout << "Dumping to File" << endl;

          ofstream myfile ("data/NNMF.nmf");
          if (myfile.is_open()) {
            for(long count = 0; count < ((totalSize) * 512); count++){

              if ((count + 1) % 512 == 0) {

                 myfile << actualArray[count] << "\n" ;

              }
              else {

                myfile << actualArray[count] << " " ;

              }
            }
            myfile.close();
          }

          // ofstream mykeyfile ("data/key.csv");
          // if (mykeyfile.is_open()) {
          //   for(long i = 0; i < MNormal; i++){
          //
          //      mykeyfile << key[i] << "\n" ;
          //
          //    }
          //
          //  }
          //   mykeyfile.close();

          }

           cout<<"NNMF.csv created successfuly"<<endl;
      }

      return 0;

    }



//method implementations


void printDeviceProperties(){
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
void printArray(uint32 * array, uint32 width){
    uint32 i;
    for (i=0;i<width;i++){
      printf("%u ", array[i]);
    }
    cout<<endl;
}

uint32* extractMartrices(TIFF* tif, string fileName){
  string newtiff = fileName.substr(0, fileName.length() - 8) + "_TP1.tif";
  TIFF* firstTimePoint = TIFFOpen(newtiff.c_str(), "w");
  if(firstTimePoint){
    tdata_t buf;
    uint32 config;

    uint32 height, width, photo;
    short samplesPerPixel, bitsPerSample;
    tsize_t scanLineSize;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

    uint32* currentTimePoint = new uint32[width*height];

    TIFFSetField(firstTimePoint, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(firstTimePoint, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(firstTimePoint, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
    TIFFSetField(firstTimePoint, TIFFTAG_BITSPERSAMPLE,bitsPerSample);
    TIFFSetField(firstTimePoint, TIFFTAG_PHOTOMETRIC, photo);
    cout<<"\nTIMEPOINT 1 .tif info:"<<endl;
    printf("width = %d\nheight = %d\nsamplesPerPixel = %d\nbitsPerSample = %d\n\n",width,height,samplesPerPixel,bitsPerSample);
    scanLineSize = TIFFScanlineSize(tif);
    buf = _TIFFmalloc(scanLineSize);
    cout<<"TIFF SCANLINE SIZE IS "<<scanLineSize<<" bits"<<endl;
    //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
    for (uint32 row = 0; row < height; row++){
      if(TIFFReadScanline(tif, buf, row, 0) != -1){
        memcpy(&currentTimePoint[row*width], buf, scanLineSize);
        if(TIFFWriteScanline(firstTimePoint, buf, row, 0) == -1){
          cout<<"ERROR WRITING SCANLINE"<<endl;
          exit(-1);
        }
      }
      else{
        cout<<"ERROR READING SCANLINE"<<endl;
        exit(-1);
      }
    }
    TIFFClose(firstTimePoint);
    _TIFFfree(buf);
    return currentTimePoint;
  }
  else{
    cout<<"COULD NOT CREATE FIRST TIMEPOINT TIFF"<<endl;
    exit(-1);
  }
}
uint32* extractMartrices(TIFF* tif, string fileName, int currentTimePoint){
  string newtiff = fileName.substr(0, fileName.length() - 8) + "_TP" + to_string(currentTimePoint) + ".tif";
  TIFF* timePoint = TIFFOpen(newtiff.c_str(), "w");
  if(timePoint){
    tdata_t buf;
    uint32 config;

    uint32 height, width, photo;
    short samplesPerPixel, bitsPerSample;
    tsize_t scanLineSize;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

    uint32* currentTimePoint = new uint32[width*height];

    TIFFSetField(timePoint, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(timePoint, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(timePoint, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
    TIFFSetField(timePoint, TIFFTAG_BITSPERSAMPLE,bitsPerSample);
    TIFFSetField(timePoint, TIFFTAG_PHOTOMETRIC, photo);
    cout<<"\nTIMEPOINT 1 .tif info:"<<endl;
    printf("width = %d\nheight = %d\nsamplesPerPixel = %d\nbitsPerSample = %d\n\n",width,height,samplesPerPixel,bitsPerSample);
    scanLineSize = TIFFScanlineSize(tif);
    buf = _TIFFmalloc(scanLineSize);
    cout<<"TIFF SCANLINE SIZE IS "<<TIFFScanlineSize(tif)<<" bits"<<endl;
    //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
    for (uint32 row = 0; row < height; row++){
      if(TIFFReadScanline(tif, buf, row, 0) != -1){
        memcpy(&currentTimePoint[row*width], buf, scanLineSize);
        if(TIFFWriteScanline(timePoint, buf, row, 0) == -1){
          cout<<"ERROR WRITING SCANLINE"<<endl;
          exit(-1);
        }
      }
      else{
        cout<<"ERROR READING SCANLINE"<<endl;
        exit(-1);
      }
    }
    TIFFClose(timePoint);
    _TIFFfree(buf);
    return currentTimePoint;
  }
  else{
    cout<<"COULD NOT CREATE FIRST TIMEPOINT TIFF"<<endl;
    exit(-1);
  }
}
uint32* extractMartrices(TIFF* tif){

  uint32 height,width;
  tdata_t buf;
  uint32 config;
  vector<uint32*> currentPlane;
  tsize_t scanLineSize;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

  uint32* currentTimePoint = new uint32[width*height];
  scanLineSize = TIFFScanlineSize(tif);
  buf = _TIFFmalloc(scanLineSize);

  //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
  for (uint32 row = 0; row < height; row++){
    if(TIFFReadScanline(tif, buf, row, 0) != -1){
      memcpy(&currentTimePoint[row*width], buf, scanLineSize);
    }
    else{
      cout<<"ERROR READING SCANLINE"<<endl;
      exit(-1);
    }

  }
  _TIFFfree(buf);
  return currentTimePoint;
}

uint32** hostTranspose(uint32** matrix, int rows, int cols){
  uint32** transposable = new uint32*[rows];
  for(int row = 0; row < rows; ++row){
    transposable[row] = new uint32[cols];
    for(int col = 0; col < cols; ++col){
      transposable[row][col] = matrix[col][row];
    }
    //cout<<"Timepoint "<<row<<" trasposed..."<<endl;

  }

  return transposable;
}

__global__ void transposeuint32Matrix(uint32* flatOrigin, uint32* flatTransposed, long Nrows, long Ncols){

  long globalID = blockIdx.x * blockDim.x + threadIdx.x;
  long pixel = globalID;
  long stride = gridDim.x * blockDim.x;
  long flatLength = Nrows * Ncols;
  long row = 0;
  long col = 0;
  uint32 currentPixelIntensity = 0;
  while(pixel < flatLength){
    row = pixel/Ncols;
    col = pixel - Ncols*row;
    flatTransposed[pixel] = flatOrigin[row + Nrows*col];
    pixel += stride;
  }

}

vector<uint32> flattenMatrix(vector<uint32*> matrix, int cols, int rows){
  vector<uint32> flat;
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      flat.push_back(matrix[r][c]);
    }
  }
  //cout<<"Matrix is flattened."<<endl;
  return flat;
}
uint32 findMin(uint32* flatMatrix, int size){
  uint32 currentMin = 0;
  for(int i = 0; i < size; ++i){
    if(currentMin > flatMatrix[i]){
      currentMin = flatMatrix[i];
    }
  }
  return currentMin;
}
__global__ void calcCa(uint32* flatMatrix, uint32 min, long size){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  int globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  long currentIndex = globalID;
  while(currentIndex < size){
    flatMatrix[globalID] = flatMatrix[globalID] - min + 1;//+1 to ensure we do not have 0 values
    currentIndex += stride;
  }
}
__global__ void fillTestMatrix(uint32* flatMatrix, long size){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  int globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  long currentIndex = globalID;
  curandState state;
  while(currentIndex < size){
    curand_init(clock64(), currentIndex, 0, &state);
    flatMatrix[currentIndex] = curand_uniform(&state);
    currentIndex += stride;
  }
}

double calculatePearsonCorrelationCoefficient(uint32 * x, uint32 * y, double lengthOfSets) {

  double * subVarianceX = new double[(int) lengthOfSets];
  double * subVarianceY = new double[(int) lengthOfSets];

  calculateSubvariance(x, lengthOfSets, subVarianceX);
  calculateSubvariance(x, lengthOfSets, subVarianceX);

  double coVarianceXY = calculateCoVariance(subVarianceX, subVarianceY, lengthOfSets);
  double standarDeviationX = calculateStandardDeviation(subVarianceX, lengthOfSets);
  double standarDeviationY = calculateStandardDeviation(subVarianceY, lengthOfSets);

  double pearsonCorrelationCoefficient = coVarianceXY / (standarDeviationX * standarDeviationY);

  delete[] subVarianceX;
  delete[] subVarianceY;

  return pearsonCorrelationCoefficient;

}

double calculateAverage(uint32 * inputSet, double lengthOfSet) {

  double average = 0;

  for (int i = 0; i < lengthOfSet; i++) {

    average = average + (double) inputSet[i];

  }

  average = average / lengthOfSet;

  return average;

}

void calculateSubvariance(uint32 * inputSet, double lengthOfSet, double * resultArray) {

  double average = calculateAverage(inputSet, lengthOfSet);

  for (int i = 0; i < lengthOfSet; i++) {

    resultArray[i] = ((double) inputSet[i]) - average;

  }

}

double calculateCoVariance(double * subVarX, double * subVarY, double lengthOfSets) {

  double coVariance = 0;

  for (int i = 0; i < lengthOfSets; i++) {

    coVariance = coVariance + (subVarX[i] * subVarY[i]);

  }

  return coVariance;

}

double calculateStandardDeviation(double * subVarX, double lengthOfSet) {

  double standarDeviation = 0;

  for (int i = 0; i < lengthOfSet; i++) {

    standarDeviation = standarDeviation + (subVarX[i] * subVarX[i]);

  }

  return standarDeviation;

}

void transposeArray(vector<uint32*> inputArray, int n, int m, uint32 * outputArray, uint32 & min, uint32 & max) {

  int outputArrayIndex = 0;

  for(unsigned i=0; i < m; i++) {

    for(unsigned j=0; j < n; j++) {

      if(inputArray[j][i] < min) {

       min = inputArray[j][i];

      }

      if(inputArray[j][i] > max) {

         max = inputArray[j][i];

      }

      outputArray[outputArrayIndex] = inputArray[j][i];
      outputArrayIndex++;

    }

  }

}
