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
#include <cfloat>
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

/*
METHOD DECLARATIONS
*/

void printDeviceProperties();
string createFourCharInt(int i);
void printArray(uint32 * array, uint32 width);
uint32* extractMartrices(TIFF* tif, string fileName);
uint32* extractMartrices(TIFF* tif);
vector<uint32> flattenMatrix(vector<uint32*> matrix, int cols, int rows);
uint32** hostTranspose(uint32** matrix, int rows, int cols);
__global__ void transposeuint32Matrix(uint32* flatOrigin, uint32* flatTransposed, long Nrows, long Ncols);
uint32 findMin(uint32* flatMatrix, int size);
__global__ void calcCa(uint32* flatMatrix, float* calcium, uint32 min, long size);
__global__ void calcFiringRate(float* frMatrix, long size, int numTimePoints);
__global__ void calcFiringRateExpanded(float* frMatrix, long size, int numTimePoints);
__global__ void fillTestMatrix(uint32* flatMatrix, long size);
void transposeArray(vector<uint32*> inputArray, int n, int m, uint32 * outputArray, uint32 & min, uint32 & max);

/*
MAIN
*/

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
          flatMatrix = extractMartrices(tif);
          flattenedTimePoints.push_back(flatMatrix);
          TIFFClose(tif);

        }
        else{
          allTifsAreGood = false;
          break;
        }
      }
      if (allTifsAreGood) {

          int NNormal = numTimePoints;
          int MNormal = (numRows*numColumns);

          cout<<"flattening"<<endl;

          uint32 min = UINT32_MAX;
          uint32 max = 0;
          uint32* temp = new uint32[MNormal*NNormal];
          int indexOfTemp = 0;
          int nonZeroCounter = 0;
          uint32* rowArray = new uint32[numColumns];
          int rowArrayIndex = 0;
          for(unsigned i=0; i < MNormal; i++) {

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
          }
          //need to delete all flattenedTimePoints arrays
          delete[] rowArray;

          uint32* actualArray = new uint32[MNormal*NNormal];
          float* firingRateArray = new float[MNormal*NNormal];
          cout << "loading arrays" << endl;

          for (long i = 0; i < MNormal*NNormal; i++) {
            //firingRateArray[i] = 0.0f;
            actualArray[i] = temp[i];

          }

          dim3 grid = {1,1,1};
          dim3 block = {1,1,1};

          if(65535 > MNormal*NNormal){
            grid.x = MNormal*NNormal;
          }
          else if(65535*1024 > MNormal*NNormal){
            grid.x = 65535;
            block.x = 1024;
            while(block.x*grid.x > MNormal*NNormal){
              block.x--;
            }
          }
          else{
            grid.x = 65535;
            block.x = 1024;
            while(grid.x*grid.y*block.x < MNormal*NNormal){
              grid.y++;
            }
          }
          cout<<"prepare for calcCa cuda kernel with min = "<<min<<",max = "<<max<<endl;
          float* firingRateArrayDevice;
          uint32* actualArrayDevice;
          CudaSafeCall(cudaMalloc((void**)&actualArrayDevice,MNormal*NNormal*sizeof(uint32)));
          CudaSafeCall(cudaMalloc((void**)&firingRateArrayDevice,MNormal*NNormal*sizeof(float)));
          CudaSafeCall(cudaMemcpy(actualArrayDevice,actualArray, MNormal*NNormal*sizeof(uint32), cudaMemcpyHostToDevice));
          CudaSafeCall(cudaMemcpy(firingRateArrayDevice,firingRateArray, MNormal*NNormal*sizeof(float), cudaMemcpyHostToDevice));
          calcCa<<<grid,block>>>(actualArrayDevice, firingRateArrayDevice, min,MNormal*NNormal);
          CudaCheckError();
          CudaSafeCall(cudaMemcpy(firingRateArray,firingRateArrayDevice, MNormal*NNormal*sizeof(float), cudaMemcpyDeviceToHost));
          for(int i = 0; i < MNormal*NNormal; ++i){
            if(!std::isfinite(firingRateArray[i])){
              cout<<"ERROR NON FINITE CALCIUM CONCENTRATION "<<firingRateArray[i]<<endl;
              exit(-1);
            }
            if(firingRateArray[i] < 0.0f){
              cout<<"ERROR NEGATIVE CALCIUM CONCENTRATION "<<firingRateArray[i]<<endl;
              exit(-1);
            }

          }
          cout<<"Executing firing rate cuda kernel"<<endl;
          calcFiringRate<<<grid,block>>>(firingRateArrayDevice, MNormal*NNormal, numTimePoints);
          CudaSafeCall(cudaMemcpy(firingRateArray,firingRateArrayDevice, MNormal*NNormal*sizeof(float), cudaMemcpyDeviceToHost));
          CudaSafeCall(cudaFree(actualArrayDevice));
          CudaSafeCall(cudaFree(firingRateArrayDevice));
          delete[] actualArray;
          cout<<"calcCa has completed applying offset"<<endl;

          float* tempCalc = new float[MNormal*NNormal];
          indexOfTemp = 0;
          int lastGoodIndex = 0;

          float *newRowArray = new float[NNormal];
          float calcMin = FLT_MAX;
          float calcMax = 0;
          cout<<"Creating key"<<endl;

          bool* key = new bool[MNormal];
          for (int i = 0; i < MNormal; i++) {

            key[i] = false;


          }

          for(unsigned i=0; i < MNormal; i++) {

            nonZeroCounter = 0;
            for(unsigned j=0; j < NNormal; j++) {

              if (firingRateArray[(NNormal*i) + j] != 0.0f){
                nonZeroCounter++;
              }
              if(!std::isfinite(firingRateArray[(NNormal*i) + j])){
                cout<<"ERROR NON FINITE NUMBER "<<firingRateArray[(NNormal*i) + j]<<endl;
                exit(-1);
              }
              if(firingRateArray[(NNormal*i) + j] < 0.0f){
                cout<<"ERROR NEGATIVE FIRIING RATE"<<endl;
                exit(-1);
              }
              newRowArray[j] = firingRateArray[(NNormal*i) + j];
            }
            // if (nonZeroCounter != 0) {
            //
            //   for (int k = 0; k < NNormal; k++) {
            //     if(newRowArray[k] < calcMin) calcMin = newRowArray[k];
            //     if(newRowArray[k] > calcMax) calcMax = newRowArray[k];
            //     tempCalc[indexOfTemp] = newRowArray[k];
            //     newRowArray[k] = 0.0f;
            //     indexOfTemp++;
            //     key[i] = true;
            //
            //   }
            //
            //   lastGoodIndex++;
            //
            // }
            // else{
            //   cout<<"EMPTY ROW FOR PIXEL "<<i<<endl;
            // }

            for (int k = 0; k < NNormal; k++) {
              if(newRowArray[k] < calcMin) calcMin = newRowArray[k];
              if(newRowArray[k] > calcMax) calcMax = newRowArray[k];
              tempCalc[indexOfTemp] = newRowArray[k];
              newRowArray[k] = 0.0f;
              indexOfTemp++;
              key[i] = true;

            }

            lastGoodIndex++;

          }
          cout << lastGoodIndex << endl;
          if(lastGoodIndex == NNormal - 1){
            cout<<"KEY CREATED BUT ALL PIXELS HAVE ATLEAST 1 NONZERO VALUE"<<endl;
          }
          cout << "MAX = "<<calcMax<<" AND MIN = "<<calcMin<<endl;

          delete[] firingRateArray;
          cout << "Dumping to File" << endl;

          ofstream myfile ("data/NNMF.nmf");
          if (myfile.is_open()) {
            for(int i = 0; i < (lastGoodIndex)*NNormal; i++){

              if ((i + 1) % 512 == 0) {

                myfile << tempCalc[i] << "\n" ;
                //myfile << (tempCalc[i] + calcMin*-1)/calcMax << "\n" ;

              }
              else {

                myfile << tempCalc[i] << " " ;
                //myfile << (tempCalc[i] + calcMin*-1)/calcMax << " " ;

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
            cout<<"NNMF.nmf created successfuly"<<endl;

          }
          else{
            cout<<"ERROR OPENING TIFF IN THIS DIRECTORY"<<endl;
            exit(-1);
          }

      }


      return 0;

    }

/*
METHOD IMPLEMENTATIONS
*/

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
  TIFF* firstTimePoint = TIFFOpen(fileName.c_str(), "w");
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

__global__ void calcCa(uint32* flatMatrix, float* calcium, uint32 min, long size){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float caConc = 0;
  float numerator = 0;
  float denominator = 0;
  float currentValue = 0;
  float dmin =  static_cast<float>(min);
  while(globalID < size){
    if(flatMatrix[globalID] != 0){
      currentValue = static_cast<float>(flatMatrix[globalID]) - dmin;
      numerator = 460*currentValue;
      denominator = (5.5*dmin) - currentValue;
      caConc = numerator/denominator;
    }
    calcium[globalID] = caConc;
    globalID += stride;
  }
}

__global__ void calcFiringRate(float* frMatrix, long size, int numTimePoints){
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float caConc = 0.0f;
  float nextCaConc = 0.0f;
  float firingRate = 0.0f;
  float tau = 0.15;
  float expValue = exp(0.0416777/tau);
  float expValuem1 = expm1(0.0416777/tau);
  float multiplier = 1/(tau*250.0f);//250 is in nm
  float numerator = 0.0f;
  int currentTimePoint = globalID % numTimePoints;
  int currentPixel = globalID/numTimePoints;
  while(globalID < size && currentTimePoint < numTimePoints - 1){
    firingRate = 0.0f;
    caConc = frMatrix[globalID];
    nextCaConc = frMatrix[globalID + 1];
    if(nextCaConc != 0.0f){//this will cause firing rate to be 0
      numerator = (nextCaConc*expValue) - caConc;
      if(numerator < 0){//currently these values will be set to 0
        printf("ERROR resulting in negative number %.9f => %.9f, %.9f, TP %d, P %d \n",numerator,caConc, nextCaConc, currentTimePoint, currentPixel);
      }
      else{
        firingRate = multiplier*numerator/expValuem1;
      }
    }
    frMatrix[globalID] = firingRate;
    if(currentTimePoint == numTimePoints - 2){//not sure this is what we want to do
      frMatrix[globalID + 1] = firingRate;
      return;
    }
    globalID += stride;
  }
}

//not implemented
__global__ void calcFiringRateExpanded(float* frMatrix, long size, int numTimePoints){

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

void updateHeightMatrix(float* heightMatrix, float* widthMatrix,
  float* uMatrix, float* sMatrix, float* vtMatrix, float* newHeightMatrix,
  int numPixels, int numTime, int numSingularValues) {

    float* widthMatrixTransposedDevice;
    float* uMatrixDevice;
    float* tempSquareMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&widthMatrixTransposedDevice, numPixels * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&uMatrixDevice, numPixels * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&tempSquareMatrixDevice, numSingularValues
      * numSingularValues * sizeof(float)));

    float* widthMatrixTransposed = new float[numPixels * numSingularValues];

    for (int i = 0; i < numPixels; i++) {

      for (int j = 0; j < numSingularValues; j++) {

        widthMatrixTransposed[j * numPixels + i] = widthMatrix[i * numSingularValues + j]

      }

    }

    CudaSafeCall(cudaMemcpy(widthMatrixTransposedDevice, widthMatrixTransposed, numPixels
      * numSingularValues * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(uMatrixDevice, uMatrix, numPixels * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(widthMatrixDevice, uMatrixDevice, tempSquareMatrixDevice, //TODO setup grid and block size
      numSingularValues, numPixels, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(uMatrixDevice));

    float* sMatrixDevice;
    float* tempSquareMatrix2Device;

    CudaSafeCall(cudaMalloc((void**)&sMatrixDevice, numSingularValues
      * numSingularValues * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&tempSquareMatrix2Device, numSingularValues
      * numSingularValues * sizeof(float)));

    CudaSafeCall(cudaMemcpy(sMatrixDevice, sMatrix, numSingularValues * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(tempSquareMatrixDevice, sMatrixDevice, tempSquareMatrix2Device, //TODO setup grid and block size
      numSingularValues, numSingularValues, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(sMatrixDevice));

    float* vtMatrixDevice;
    float* numeratorDevice;

    CudaSafeCall(cudaMalloc((void**)&vtMatrixDevice, numSingularValues
      * numTime * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&numeratorDevice, numSingularValues
      * numTime * sizeof(float)));

    CudaSafeCall(cudaMemcpy(vtMatrixDevice, vtMatrix, numSingularValues * numTime
      * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(tempSquareMatrix2Device, vtMatrixDevice, numeratorDevice, //TODO setup grid and block size
      numSingularValues, numSingularValues, numTime);

    CudaCheckError();

    CudaSafeCall(cudaFree(tempSquareMatrix2Device));
    CudaSafeCall(cudaFree(vtMatrixDevice));

    float* widthMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&widthMatrixDevice, numPixels
      * numSingularValues * sizeof(float)));

    CudaSafeCall(cudaMemcpy(widthMatrixDevice, widthMatrix, numPixels * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(widthMatrixTransposedDevice, widthMatrixDevice, tempSquareMatrixDevice, //TODO setup grid and block size
      numSingularValues, numPixels, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(widthMatrixTransposed));
    CudaSafeCall(cudaFree(widthMatrixDevice));

    float* heightMatrixDevice;
    float* denominatorDevice;

    CudaSafeCall(cudaMalloc((void**)&heightMatrixDevice, numSingularValues
      * numTime * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&newHeightMatrixDevice, numSingularValues
      * numTime * sizeof(float)));

    CudaSafeCall(cudaMemcpy(heightMatrixDevice, heightMatrix, numSingularValues *
      numTime * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(tempSquareMatrixDevice, heightMatrixDevice, //TODO setup grid and block size
      denominatorDevice, numSingularValues, numSingularValues, numTime);

    CudaCheckError();

    CudaSafeCall(cudaFree(tempSquareMatrixDevice));

    applyScalar<<<grid,block>>>(heightMatrixDevice, numeratorDevice, //TODO setup grid and block size
      denominatorDevice, numSingularValues, numTime);

    CudaCheckError();

    CudaSafeCall(cudaMemcpy(heightMatrix, heightMatrixDevice, numSingularValues*
      * numTime * sizeof(float), cudaMemcpyDeviceToHost));

    CudaCheckError();

    CudaSafeCall(cudaFree(heightMatrixDevice));
    CudaSafeCall(cudaFree(numeratorDevice));
    CudaSafeCall(cudaFree(denominatorDevice));

    delete[] widthMatrixTransposed;

  }

void updateWidthMatrix(float* heightMatrix, float* widthMatrix,
  float* uMatrix, float* sMatrix, float* vtMatrix, float* newHeightMatrix,
  int numPixels, int numTime, int numSingularValues) {

    float* heightMatrixTransposedDevice;
    float* vtMatrixDevice;
    float* tempSquareMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&heightMatrixTransposedDevice, numSingularValues
      * numTime * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&vtMatrixDevice, numSingularValues * numTime
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&tempSquareMatrixDevice, numSingularValues
      * numSingularValues * sizeof(float)));

    float* heightMatrixTransposed = new float[numSingularValues * numTime];

    for (int i = 0; i < numSingularValues; i++) {

      for (int j = 0; j < numTime; j++) {

        heightMatrixTransposed[j * numSingularValues + i] = widthMatrix[i * numTime + j]

      }

    }

    CudaSafeCall(cudaMemcpy(heightMatrixTransposedDevice, heighMatrixTransposed,
      numSingularValues * numTime * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(vtMatrixDevice, vtMatrix, numSingularValues * numTime
      * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(vtMatrixDevice, heightMatrixTransposedDevice,
      tempSquareMatrixDevice, numSingularValues, numTime, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(vtMatrixDevice));

    float* sMatrixDevice;
    float* tempSquareMatrix2Device;

    CudaSafeCall(cudaMalloc((void**)&sMatrixDevice, numSingularValues
      * numSingularValues * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&tempSquareMatrix2Device, numSingularValues
      * numSingularValues * sizeof(float)));

    CudaSafeCall(cudaMemcpy(sMatrixDevice, sMatrix, numSingularValues
      * numSingularValues * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(sMatrixDevice, tempSquareMatrixDevice,
      tempSquareMatrix2Device, numPixels, numSingularValues, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(sMatrixDevice));

    float* uMatrixDevice;
    float* numeratorDevice;

    CudaSafeCall(cudaMalloc((void**)&uMatrixDevice, numPixels * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&numeratorDevice, numPixels
      * numSingularValues * sizeof(float)));

    CudaSafeCall(cudaMemcpy(uMatrixDevice, uMatrix, numPixels * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(uMatrixDevice, tempSquareMatrix2Device,
      numeratorDevice, numPixels, numSingularValues, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(uMatrixDevice));
    CudaSafeCall(cudaFree(tempSquareMatrix2Device));

    float* heightMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&heightMatrix, numSingularValues * numTime
      * sizeof(float)));

    CudaSafeCall(cudaMemcpy(heightMatrixDevice, heightMatrix, numSingularValues
      * numTime * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(heightMatrixDevice, heightMatrixTransposedDevice,
      tempSquareMatrixDevice, numSingularValues, numTime, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(heightMatrixTransposedDevice));
    CudaSafeCall(cudaFree(heightMatrixDevice));

    float* widthMatrixDevice;
    float* denominatorDevice;

    CudaSafeCall(cudaMalloc((void**)&widthMatrixDevice, numPixels * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&denominatorDevice, numPixels * numSingularValues
      * sizeof(float)));

    CudaSafeCall(cudaMemcpy(widthMatrixDevice, widthMatrix, numPixels * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    multiplyMatrices<<<grid,block>>>(widthMatrixDevice, tempSquareMatrixDevice,
      denominatorDevice, numPixels, numSingularValues, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(tempSquareMatrixDevice));

    applyScalar<<<grid,block>>>(widthMatrixDevice, numeratorDevice, //TODO setup grid and block size
      denominatorDevice, numPixels, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaMemcpy(widthMatrix, widthMatrixDevice, numPixels *
      * numSingularValues * sizeof(float), cudaMemcpyDeviceToHost));

    CudaCheckError();

    CudaSafeCall(cudaFree(widthMatrixDevice));
    CudaSafeCall(cudaFree(numeratorDevice));
    CudaSafeCall(cudaFree(denominatorDevice));

    delete[] heightMatrixTransposed;

}

__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, int diffDimA,
   int comDim, int diffDimB) {

     int blockID = blockIdx.y * gridDim.x + blockIdx.x;
     int globalID = blockID * blockDim.x + threadIdx.x;
     long currentIndex = globalID;

     if (currentIndex < (diffDimA * diffDimB)) {

       int iIndex = currentIndex / diffDimB;
       int jIndex = currentIndex % diffDimB;

       int sum = 0;

       for (int k = 0; k < comDim; k++) {

         sum = sum + (matrixA[iIndex * comDim + k] * matrixB[k * diffDimB + jIndex])

       }

       matrixC[iIndex * diffDimB + jIndex] = sum;

     }

   }

__global__ void applyScalar(float* targetMatrix, float* numerator, float* denominator,
    int numRows, int numCols) {

    int blockID = blockIdx.y * gridDim.x + blockIdx.x;
    int globalID = blockID * blockDim.x + threadIdx.x;
    long currentIndex = globalID;

    if (currentIndex < (diffDimA * diffDimB)) {

      int iIndex = currentIndex / diffDimB;
      int jIndex = currentIndex % diffDimB;

      targetMatrix[iIndex * numCols + jIndex] = targetMatrix[iIndex * numCols + jIndex]
        * (numerator[iIndex * numCols + jIndex] / denominator[iIndex * numCols + jIndex]);

    }

  }
