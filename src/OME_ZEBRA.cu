#include <stdio.h>
#include <stdlib.h>
#include <string>
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
#include <cstdlib>
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
string createFourCharInt(int i);
void printArray(uint32 * array, uint32 width);
uint32* extractMartrices(TIFF* tif, string fileName);
uint32* extractMartrices(TIFF* tif, string fileName, int currentTimePoint);
uint32* extractMartrices(TIFF* tif);
vector<uint32> flattenMatrix(vector<uint32*> matrix, int cols, int rows);
uint32** hostTranspose(uint32** matrix, int rows, int cols);
__global__ void transposeuint32Matrix(uint32* flatOrigin, uint32* flatTransposed, long Nrows, long Ncols);
uint32 findMin(uint32* flatMatrix, int size);
__global__ void calcCa(uint32* flatMatrix, float* calcium, uint32 min, uint32 max, long size);
__global__ void calcFiringRate(float* frMatrix, long size, int numTimePoints);
__global__ void fillTestMatrix(uint32* flatMatrix, long size);
void transposeArray(vector<uint32*> inputArray, int n, int m, uint32 * outputArray, uint32 & min, uint32 & max);


int main(int argc, char *argv[]) {

    if(argc != 3) {
      cout << "Usage: ./exe <file> <# of time points>";
      return 1;
    }
    else {

      vector<uint32*> flattenedTimePoints;
      string baseName = argv[1];
      int numTimePoints = atoi(argv[2]);
      if(numTimePoints == 0){
        cout<<"ERROR INVALID TIMEPOINTS"<<endl;
        exit(-1);
      }
      bool allTifsAreGood = true;
      uint32 numColumns;
      uint32 numRows;
      string currentTif;
      for(int i = 0; i < numTimePoints; ++i){

        currentTif = "data/registeredOMEs/" + baseName + "/" +
        baseName + ".ome" + createFourCharInt(i) + ".tif";

        TIFF* tif = TIFFOpen(currentTif.c_str(), "r");

        if (tif) {
          if(i == 0){
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &numColumns);
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &numRows);
          }
          uint32 tempCol;
          uint32 tempRow;
          cout<<currentTif<<" IS OPENED"<<endl;
          TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &tempCol);
          TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &tempRow);
          if(numRows != tempRow || numColumns != tempCol){
            cout<<"ERROR NOT ALL TIFFS ARE THE SAME LENGTH"<<endl;
            exit(-1);
          }

          uint32* flatMatrix = new uint32[numRows*numColumns];
          if(i == 0){
            flatMatrix = extractMartrices(tif, baseName);
          }
          else{
            flatMatrix = extractMartrices(tif);
          }
          flattenedTimePoints.push_back(flatMatrix);
          TIFFClose(tif);

        }
        else{
          allTifsAreGood = false;
          break;
        }
      }
      if (allTifsAreGood) {

          cout<<"Creating key"<<endl;
          int NNormal = numTimePoints;
          int MNormal = (numRows*numColumns);

          bool* key = new bool[MNormal];
          for (int i = 0; i < MNormal; i++) {

            key[i] = false;

          }


          uint32* temp;
          uint32 min = 4294967295;
          uint32 max = 0;
          temp = new uint32[MNormal*NNormal];
          int indexOfTemp = 0;
          int nonZeroCounter = 0;
          uint32* rowArray = new uint32[numColumns];
          int rowArrayIndex = 0;
          int lastGoodIndex = 0;
          bool allRealRows = false;
          for(unsigned i=0; i < MNormal; i++) {

            allRealRows = false;
            nonZeroCounter = 0;
            rowArrayIndex = 0;
            for(unsigned j=0; j < NNormal; j++) {

              if (flattenedTimePoints[j][i] != 0){

                nonZeroCounter++;
                if(flattenedTimePoints[j][i] < min) min = flattenedTimePoints[j][i];
                if(flattenedTimePoints[j][i] > max) max = flattenedTimePoints[j][i];

              }

              rowArray[rowArrayIndex] = flattenedTimePoints[j][i];
              rowArrayIndex++;

            }

            for (int k = 0; k < NNormal; k++) {

              temp[indexOfTemp] = rowArray[k];
              rowArray[k] = 0;
              indexOfTemp++;

            }

            lastGoodIndex++;
            allRealRows = true;

          }

          cout << lastGoodIndex << endl;
          long minimizedSize = lastGoodIndex*512;
          uint32* actualArray = new uint32[minimizedSize];
          float* firingRateArray = new float[minimizedSize];
          cout << "loading arrays based on key" << endl;

          for (long i = 0; i < minimizedSize; i++) {
            firingRateArray[i] = 0.0f;
            actualArray[i] = temp[i];

          }

          dim3 grid = {1,1,1};
          dim3 block = {1,1,1};

          if(65535 > minimizedSize){
            grid.x = minimizedSize;
          }
          else if(65535*1024 > minimizedSize){
            grid.x = 65535;
            block.x = 1024;
            while(block.x*grid.x > minimizedSize){
              block.x--;
            }
          }
          else{
            grid.x = 65535;
            block.x = 1024;
            while(grid.x*grid.y*block.x < minimizedSize){
              grid.y++;
            }
          }
          cout<<"prepare for calcCa cuda kernel with min = "<<min<<",max = "<<max<<endl;
          float* firingRateArrayDevice;
          uint32* actualArrayDevice;
          CudaSafeCall(cudaMalloc((void**)&actualArrayDevice,minimizedSize*sizeof(uint32)));
          CudaSafeCall(cudaMalloc((void**)&firingRateArrayDevice,minimizedSize*sizeof(float)));
          CudaSafeCall(cudaMemcpy(actualArrayDevice,actualArray, minimizedSize*sizeof(uint32), cudaMemcpyHostToDevice));
          CudaSafeCall(cudaMemcpy(firingRateArrayDevice,firingRateArray, minimizedSize*sizeof(float), cudaMemcpyHostToDevice));
          calcCa<<<grid,block>>>(actualArrayDevice, firingRateArrayDevice, min, max, minimizedSize);
          CudaCheckError();
          cout<<"Executing firing rate cuda kernel"<<endl;
          calcFiringRate<<<grid,block>>>(firingRateArrayDevice, minimizedSize, numTimePoints);
          CudaSafeCall(cudaMemcpy(firingRateArray,firingRateArrayDevice, minimizedSize*sizeof(float), cudaMemcpyDeviceToHost));
          CudaSafeCall(cudaFree(actualArrayDevice));
          CudaSafeCall(cudaFree(firingRateArrayDevice));
          cout<<"calcCa has completed applying offset"<<endl;

          float* tempCalc = new float[minimizedSize];
          indexOfTemp = 0;
          lastGoodIndex = 0;
          float * newRowArray = new float[NNormal];
          float calcMin = 999999.0;
          float calcMax = -1024.0;

          for(unsigned i=0; i < MNormal; i++) {

            nonZeroCounter = 0;
            rowArrayIndex = 0;

            for(unsigned j=0; j < NNormal; j++) {

              if (firingRateArray[(NNormal*i) + j] != 0){

                nonZeroCounter++;
                if(firingRateArray[(NNormal*i) + j] < calcMin) calcMin = firingRateArray[(NNormal*i) + j];
                if(firingRateArray[(NNormal*i) + j] > calcMax) calcMax = firingRateArray[(NNormal*i) + j];

              }

              newRowArray[rowArrayIndex] = firingRateArray[(NNormal*i) + j];
              rowArrayIndex++;

            }
            if (nonZeroCounter != 0) {

              for (int k = 0; k < NNormal; k++) {

                tempCalc[indexOfTemp] = newRowArray[k];
                rowArray[k] = 0;
                indexOfTemp++;
                key[i] = true;

              }

              lastGoodIndex++;

            }

          }

          cout << lastGoodIndex << endl;

          delete[] firingRateArray;

          cout << "Test" << endl;

          float* newFiringRateArray = new float[lastGoodIndex*NNormal];

          for (int i = 0; i < (lastGoodIndex+1)*NNormal; i++) {

            newFiringRateArray[i] = (tempCalc[i] + calcMin*-1)/calcMax;

            if (i % NNormal == 0) {

              cout << lastGoodIndex << " " << i/512 << endl;

            }

          }

          cout << "Dumping to File" << endl;

          ofstream myfile ("data/NNMF.nmf");
          if (myfile.is_open()) {
            for(int i = 0; i < (lastGoodIndex+1)*NNormal; i++){

              if (i % NNormal == 0) {

                cout << lastGoodIndex << " " << i/512 << endl;

              }

              if ((i + 1) % 512 == 0) {

                //myfile << actualArray[count] << "\n" ;
                myfile << newFiringRateArray[i] << "\n" ;

              }
              else {

                //myfile << actualArray[count] << " " ;
                myfile << newFiringRateArray[i] << " " ;

              }
            }
            myfile.close();
          }

          cout << "done" << endl;

          ofstream mykeyfile ("data/key.csv");
          if (mykeyfile.is_open()) {
            for(long i = 0; i < MNormal; i++){

               mykeyfile << key[i] << "\n" ;

             }

           }
            mykeyfile.close();
            cout<<"NNMF.csv created successfuly"<<endl;

          }
          else{
            cout<<"ERROR OPENING TIFF IN THIS DIRECTORY"<<endl;
            exit(-1);
          }

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
string createFourCharInt(int i){
  string strInt;
  if(i < 10){
    strInt = "000" + to_string(i);
  }
  else if(i < 100){
    strInt = "00" + to_string(i);
  }
  else if(i < 1000){
    strInt = "0" + to_string(i);
  }
  else{
    strInt = to_string(i);
  }
  return strInt;
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
  TIFF* currentDir = TIFFOpen(newtiff.c_str(), "w");
  if(currentDir){
    tdata_t buf;

    uint32 height, width, photo;
    short samplesPerPixel, bitsPerSample;
    tsize_t scanLineSize;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

    uint32* timePoint = new uint32[width*height];

    TIFFSetField(currentDir, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(currentDir, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(currentDir, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
    TIFFSetField(currentDir, TIFFTAG_BITSPERSAMPLE,bitsPerSample);
    TIFFSetField(currentDir, TIFFTAG_PHOTOMETRIC, photo);
    cout<<"\nTIMEPOINT "<<currentTimePoint<<" .tif info:"<<endl;
    printf("width = %d\nheight = %d\nsamplesPerPixel = %d\nbitsPerSample = %d\n\n",width,height,samplesPerPixel,bitsPerSample);
    scanLineSize = TIFFScanlineSize(tif);
    buf = _TIFFmalloc(scanLineSize);
    cout<<"TIFF SCANLINE SIZE IS "<<scanLineSize<<" bits"<<endl;
    //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
    for (uint32 row = 0; row < height; row++){
      if(TIFFReadScanline(tif, buf, row, 0) != -1){
        memcpy(&timePoint[row*width], buf, scanLineSize);
        if(TIFFWriteScanline(currentDir, buf, row, 0) == -1){
          cout<<"ERROR WRITING SCANLINE"<<endl;
          exit(-1);
        }
      }
      else{
        cout<<"ERROR READING SCANLINE"<<endl;
        exit(-1);
      }
    }
    TIFFClose(currentDir);
    _TIFFfree(buf);
    return timePoint;
  }
  else{
    cout<<"COULD NOT CREATE FIRST TIMEPOINT TIFF"<<endl;
    exit(-1);
  }
}
uint32* extractMartrices(TIFF* tif){

  uint32 height,width;
  tdata_t buf;
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

__global__ void calcCa(uint32* flatMatrix, float* calcium, uint32 min, uint32 max, long size){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float caConc = 0;
  float currentIntensity = 0;
  while(globalID < size){
    currentIntensity = (float) flatMatrix[globalID];
    caConc = (currentIntensity - min)/(max - currentIntensity);
    calcium[globalID] = 3.16227766e-7*caConc;
    globalID += stride;
  }
}

__global__ void calcFiringRate(float* frMatrix, long size, int numTimePoints){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float caConc = 0.0f;
  //float caConc = 1.0f;
  float nextCaConc = 0.0f;
  //float nextCaConc = 1.0f;
  float firingRate = 0.0f;
  float tau = 0.15;
  float deltaT = 0.0416777;
  float alpha = 250.0f;
  long numPixels = size/numTimePoints;
  int currentTimePoint = globalID % numTimePoints;
  while(globalID < size && currentTimePoint < numTimePoints - 1){
    firingRate = 0.0f;
    caConc = frMatrix[globalID];
    nextCaConc =   frMatrix[globalID + 1];
    if(nextCaConc != 0.0f) firingRate = ((nextCaConc*expf(deltaT/tau))-caConc)/(expm1f(deltaT/tau)*tau*alpha);
    frMatrix[globalID] = firingRate;
    //if(currentTimePoint == numTimePoints - 2){//not sure this is what we want to do
    //  frMatrix[globalID + 1] = firingRate;
    //  return;
    //}
    globalID += stride;
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
