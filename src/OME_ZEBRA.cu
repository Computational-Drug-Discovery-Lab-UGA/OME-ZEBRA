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
#include <limits>

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
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}

//TODO get correct firing rate equation

string createFourCharInt(int i);
uint32* extractMartrices(TIFF* tif, string fileName);
uint32* extractMartrices(TIFF* tif);

__global__ void calcCa(uint32* flatMatrix, float* calcium, uint32 min, long size);
__global__ void calcFiringRate(float* frMatrix, long size, int numTimePoints);

__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, long diffDimA,
   long comDim, long diffDimB);
__global__ void applyScalar(float* targetMatrix, float* numerator, float* denominator,
    long numRows, long numCols);
__global__ void calculateLoss(float* originalMatrix, float* newMatrix, long numRows, long numCols, float* loss);

void updateWidthMatrix(float* heightMatrix, float* widthMatrix,
  float* uMatrix, float* sMatrix, float* vtMatrix, float* newWidthMatrix,
  long numPixels, long numTime, long numSingularValues);
void updateHeightMatrix(float* heightMatrix, float* widthMatrix,
  float* uMatrix, float* sMatrix, float* vtMatrix, float* newHeightMatrix,
  long numPixels, long numTime, long numSingularValues);
void NMF(float* heightMatrix, float* widthMatrix, float* uMatrix,
  float* sMatrix, float* vtMatrix, float* originalMatrix, long numPixels, long numTime,
  long numSingularValues, float targetLoss);
float findA(float* uMatrix, float* sMatrix, float* vtMatrix,
  long numPixels, long numTime, long numSingularValues);

int main(int argc, char *argv[]) {

    if(argc != 3) {
      cout << "Usage: ./exe <file> <# of time points>";
      return 1;
    }
    else {
      dim3 grid = {1,1,1};
      dim3 block = {1,1,1};

      vector<uint32*> flattenedTimePoints;
      string baseName = argv[1];
      int numTimePoints = atoi(argv[2]);
      if(numTimePoints == 0){
        cout<<"ERROR INVALID TIMEPOINTS"<<endl;
        exit(-1);
      }
      bool allTifsAreGood = true;
      // string currentTif;
      // uint32 numRows, numColumns;
      // for(int i = 0; i < numTimePoints; ++i){
      //
      //   currentTif = "data/registeredOMEs/" + baseName + "/" +
      //   baseName + ".ome" + createFourCharInt(i) + ".tif";
      //
      //   TIFF* tif = TIFFOpen(currentTif.c_str(), "r");
      //
      //   if (tif) {
      //     if(i == 0){
      //       TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &numColumns);
      //       TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &numRows);
      //     }
      //     uint32 tempCol;
      //     uint32 tempRow;
      //     cout<<currentTif<<" IS OPENED"<<endl;
      //     TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &tempCol);
      //     TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &tempRow);
      //     if(numRows != tempRow || numColumns != tempCol){
      //       cout<<"ERROR NOT ALL TIFFS ARE THE SAME LENGTH"<<endl;
      //       exit(-1);
      //     }
      //
      //     uint32* flatMatrix = new uint32[numRows*numColumns];
      //     flatMatrix = extractMartrices(tif);
      //     flattenedTimePoints.push_back(flatMatrix);
      //     TIFFClose(tif);
      //
      //   }
      //   else{
      //     allTifsAreGood = false;
      //     break;
      //   }
      // }
      if (allTifsAreGood) {
          // int NNormal = numTimePoints;
          // int MNormal = (numRows*numColumns);
          //
          // cout<<"flattening"<<endl;
          //
          // uint32 min = UINT32_MAX;
          // uint32 max = 0;
          // uint32* temp = new uint32[MNormal*NNormal];
          // int indexOfTemp = 0;
          // int nonZeroCounter = 0;
          // uint32* rowArray = new uint32[numColumns];
          // int rowArrayIndex = 0;
          // for(unsigned i=0; i < MNormal; i++) {
          //
          //   nonZeroCounter = 0;
          //   rowArrayIndex = 0;
          //   for(unsigned j=0; j < NNormal; j++) {
          //     if (flattenedTimePoints[j][i] != 0){
          //       nonZeroCounter++;
          //       if(flattenedTimePoints[j][i] < min) min = flattenedTimePoints[j][i];
          //       if(flattenedTimePoints[j][i] > max) max = flattenedTimePoints[j][i];
          //     }
          //
          //     rowArray[rowArrayIndex] = flattenedTimePoints[j][i];
          //     rowArrayIndex++;
          //   }
          //   for (int k = 0; k < NNormal; k++) {
          //
          //     temp[indexOfTemp] = rowArray[k];
          //     rowArray[k] = 0;
          //     indexOfTemp++;
          //
          //   }
          // }
          // //need to delete all flattenedTimePoints arrays
          // delete[] rowArray;
          //
          // uint32* actualArray = new uint32[MNormal*NNormal];
          // float* firingRateArray = new float[MNormal*NNormal];
          // cout << "loading arrays" << endl;
          //
          // for (long i = 0; i < MNormal*NNormal; i++) {
          //   //firingRateArray[i] = 0.0f;
          //   actualArray[i] = temp[i];
          //
          // }
          //
          // if(65535 > MNormal*NNormal){
          //   grid.x = MNormal*NNormal;
          // }
          // else if(65535*1024 > MNormal*NNormal){
          //   grid.x = 65535;
          //   block.x = 1024;
          //   while(block.x*grid.x > MNormal*NNormal){
          //     block.x--;
          //   }
          // }
          // else{
          //   grid.x = 65535;
          //   block.x = 1024;
          //   while(grid.x*grid.y*block.x < MNormal*NNormal){
          //     grid.y++;
          //   }
          // }
          // cout<<"prepare for calcCa cuda kernel with min = "<<min<<",max = "<<max<<endl;
          // float* firingRateArrayDevice;
          // uint32* actualArrayDevice;
          // CudaSafeCall(cudaMalloc((void**)&actualArrayDevice,MNormal*NNormal*sizeof(uint32)));
          // CudaSafeCall(cudaMalloc((void**)&firingRateArrayDevice,MNormal*NNormal*sizeof(float)));
          // CudaSafeCall(cudaMemcpy(actualArrayDevice,actualArray, MNormal*NNormal*sizeof(uint32), cudaMemcpyHostToDevice));
          // CudaSafeCall(cudaMemcpy(firingRateArrayDevice,firingRateArray, MNormal*NNormal*sizeof(float), cudaMemcpyHostToDevice));
          // calcCa<<<grid,block>>>(actualArrayDevice, firingRateArrayDevice, min,MNormal*NNormal);
          // CudaCheckError();
          // CudaSafeCall(cudaMemcpy(firingRateArray,firingRateArrayDevice, MNormal*NNormal*sizeof(float), cudaMemcpyDeviceToHost));
          // for(int i = 0; i < MNormal*NNormal; ++i){
          //   if(!std::isfinite(firingRateArray[i])){
          //     cout<<"ERROR NON FINITE CALCIUM CONCENTRATION "<<firingRateArray[i]<<endl;
          //     exit(-1);
          //   }
          //   if(firingRateArray[i] < 0.0f){
          //     cout<<"ERROR NEGATIVE CALCIUM CONCENTRATION "<<firingRateArray[i]<<endl;
          //     exit(-1);
          //   }
          //   firingRateArray[i] += 1;
          // }
          // // cout<<"Executing firing rate cuda kernel"<<endl;
          // // calcFiringRate<<<grid,block>>>(firingRateArrayDevice, MNormal*NNormal, numTimePoints);
          // // CudaSafeCall(cudaMemcpy(firingRateArray,firingRateArrayDevice, MNormal*NNormal*sizeof(float), cudaMemcpyDeviceToHost));
          // CudaSafeCall(cudaFree(actualArrayDevice));
          // CudaSafeCall(cudaFree(firingRateArrayDevice));
          // delete[] actualArray;
          // cout<<"calcCa has completed applying offset"<<endl;
          //
          // float* tempCalc = new float[MNormal*NNormal];
          // indexOfTemp = 0;
          // int lastGoodIndex = 0;
          //
          // float *newRowArray = new float[NNormal];
          // float calcMin = FLT_MAX;
          // float calcMax = 0;
          // cout<<"Creating key"<<endl;
          //
          // bool* key = new bool[MNormal];
          // for (int i = 0; i < MNormal; i++) {
          //
          //   key[i] = false;
          //
          //
          // }
          //
          // for(unsigned i=0; i < MNormal; i++) {
          //
          //   nonZeroCounter = 0;
          //   for(unsigned j=0; j < NNormal; j++) {
          //
          //     if (firingRateArray[(NNormal*i) + j] != 0.0f){
          //       nonZeroCounter++;
          //     }
          //     if(!std::isfinite(firingRateArray[(NNormal*i) + j])){
          //       cout<<"ERROR NON FINITE NUMBER "<<firingRateArray[(NNormal*i) + j]<<endl;
          //       exit(-1);
          //     }
          //     if(firingRateArray[(NNormal*i) + j] < 0.0f){
          //       cout<<"ERROR NEGATIVE FIRIING RATE"<<endl;
          //       exit(-1);
          //     }
          //     newRowArray[j] = firingRateArray[(NNormal*i) + j];
          //   }
          //   // if (nonZeroCounter != 0) {
          //   //
          //   //   for (int k = 0; k < NNormal; k++) {
          //   //     if(newRowArray[k] < calcMin) calcMin = newRowArray[k];
          //   //     if(newRowArray[k] > calcMax) calcMax = newRowArray[k];
          //   //     tempCalc[indexOfTemp] = newRowArray[k];
          //   //     newRowArray[k] = 0.0f;
          //   //     indexOfTemp++;
          //   //     key[i] = true;
          //   //
          //   //   }
          //   //
          //   //   lastGoodIndex++;
          //   //
          //   // }
          //   // else{
          //   //   cout<<"EMPTY ROW FOR PIXEL "<<i<<endl;
          //   // }
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
          // cout << lastGoodIndex << endl;
          // if(lastGoodIndex == NNormal - 1){
          //   cout<<"KEY CREATED BUT ALL PIXELS HAVE ATLEAST 1 NONZERO VALUE"<<endl;
          // }
          // cout << "MAX = "<<calcMax<<" AND MIN = "<<calcMin<<endl;
          //
          // delete[] firingRateArray;
          // cout << "Dumping to File" << endl;
          //
          // ofstream myfile ("data/NNMF.nmf");
          // if (myfile.is_open()) {
          //   for(int i = 0; i < (lastGoodIndex)*NNormal; i++){
          //
          //     if ((i + 1) % 512 == 0) {
          //       myfile << tempCalc[i] << "\n" ;
          //     }
          //     else {
          //       myfile << tempCalc[i] << " " ;
          //     }
          //   }
          //   myfile.close();
          // }
          //
          // cout << "done" << endl;
          //
          // ofstream mykeyfile ("data/key.csv");
          // if (mykeyfile.is_open()) {
          //   for(long i = 0; i < MNormal; i++){
          //
          //      mykeyfile << key[i] << "\n" ;
          //
          //    }
          //
          // }
          //  mykeyfile.close();
          //  cout<<"NNMF.nmf created successfuly"<<endl;
          //  exit(0);


            long numSingularValues = 100;
            long numPixels = 524288;
            long numTime = 512;

            std::cout << "Loading sMatrix" << '\n';

            float* sMatrix = new float[numSingularValues * numSingularValues];

            std::fstream sMatrixFile("data/sMatrix.txt", std::ios_base::in);

            float singularValue;
            int indexOfSingularValue = 0;
            int colCount = 0;
            long rowCount = 0;

            while (sMatrixFile >> singularValue) {

                if (indexOfSingularValue == colCount) {

                  sMatrix[rowCount * numSingularValues + indexOfSingularValue] = singularValue;
                  indexOfSingularValue++;
                  colCount++;

                }
                else if (colCount < indexOfSingularValue) {

                  for (int i = colCount; i < indexOfSingularValue; i++) {

                    sMatrix[rowCount * numSingularValues + colCount] = 0;
                    colCount++;

                  }

                }
                else if (colCount < numSingularValues) {

                  for (int i = colCount; i < numSingularValues; i++) {

                    sMatrix[rowCount * numSingularValues + colCount] = 0;
                    colCount++;

                  }

                  rowCount++;
                  colCount = 0;

                }

            }

            std::cout << "Loading uMatrix" << '\n';

            std::fstream uMatrixFile("data/uMatrix.txt", std::ios_base::in);

            float* uMatrix = new float[numPixels * numSingularValues];

            float uValue;
            int indexOfuMatrix = 0;
            int numUZero = 0;
            while (uMatrixFile >> uValue) {
                if(uValue == 0.0f) ++numUZero;

                uMatrix[indexOfuMatrix] = uValue;
                indexOfuMatrix++;

            }

            std::cout << "Loading vtMatrix" << '\n';

            std::fstream vtMatrixFile("data/vtMatrix.txt", std::ios_base::in);

            float* vtMatrix = new float[numSingularValues * numTime];

            float vtValue;
            int indexOfvtMatrix = 0;
            int numVtZero = 0;
            while (vtMatrixFile >> vtValue) {
                if(vtValue == 0.0f) ++numVtZero;
                vtMatrix[indexOfvtMatrix] = vtValue;
                indexOfvtMatrix++;

            }


            cout << "Executing NNMF" << endl;

            float A = findA(uMatrix, sMatrix, vtMatrix,
              numPixels, numTime, numSingularValues);

            numSingularValues++;

            float* newSMatrix = new float[numSingularValues * numSingularValues];

            for (long i = 0; i < numSingularValues; i++) {

              if (i < (numSingularValues - 1)) {

                for (long j = 0; j < numSingularValues; j++) {

                  if (j < (numSingularValues - 1)) {

                    newSMatrix[numSingularValues * i + j] = sMatrix[(numSingularValues-1) * i + j];

                  }
                  else {

                    newSMatrix[numSingularValues * i + j] = 0.0;

                  }

                }

              }
              else {

                for (long j = 0; j < (numSingularValues - 1); j++) {

                  newSMatrix[numSingularValues * i + j] = 0.0;

                }

                newSMatrix[numSingularValues * numSingularValues - 1] = A;

              }

            }

            float* newUMatrix = new float[numPixels * numSingularValues];

            for (long i = 0; i < numPixels; i++) {

              for (long j = 0; j < (numSingularValues - 1); j++) {

                newUMatrix[numSingularValues * i + j] = uMatrix[(numSingularValues - 1) * i + j];

              }

              newUMatrix[numSingularValues * i + (numSingularValues - 1)] = 1.0;

            }

            float* newVTMatrix = new float[numSingularValues * numTime];

            for (long i = 0; i < (numSingularValues - 1); i++) {

              for (long j = 0; j < numTime; j++) {

                newVTMatrix[numTime * i + j] = vtMatrix[numTime * i + j];

              }

            }

            for (long j = 0; j < numTime; j++) {

              newVTMatrix[numTime * (numSingularValues - 1) + j] = 1.0;

            }

            float* heightMatrix = new float[numSingularValues * numTime];

            for (long i = 0; i < (numSingularValues * numTime); i++) {

              heightMatrix[i] = 1;

            }

            float* widthMatrix = new float[numPixels * numSingularValues];

            for (long i = 0; i < (numPixels * numSingularValues); i++) {

              widthMatrix[i] = 1;

            }

            delete[] sMatrix;
            delete[] uMatrix;
            delete[] vtMatrix;

            float targetLoss = .3;

            cout << "getting original matrix U*S*Vt" << endl;

            long a = 0;
            long b = 0;

            float* uMatrixDevice;
            float* sMatrixDevice;
            float* tempMatrixDevice;

            CudaSafeCall(cudaMalloc((void**)&uMatrixDevice, numPixels
              * numSingularValues * sizeof(float)));
            CudaSafeCall(cudaMalloc((void**)&sMatrixDevice, numSingularValues
              * numSingularValues * sizeof(float)));
            CudaSafeCall(cudaMalloc((void**)&tempMatrixDevice, numPixels
              * numSingularValues * sizeof(float)));

            CudaSafeCall(cudaMemcpy(uMatrixDevice, newUMatrix, numPixels
              * numSingularValues * sizeof(float), cudaMemcpyHostToDevice));
            CudaSafeCall(cudaMemcpy(sMatrixDevice, newSMatrix, numSingularValues
              * numSingularValues * sizeof(float), cudaMemcpyHostToDevice));

            a = numPixels;
            b = numSingularValues;

            if(65535 > a*b){
              grid.x = a*b;
            }
            else if(65535*1024 > a*b){
              grid.x = 65535;
              block.x = 1024;
              while(block.x*grid.x > a*b){
                block.x--;
              }
            }
            else{
              grid.x = 65535;
              block.x = 1024;
              while(grid.x*grid.y*block.x < a*b){
                grid.y++;
              }
            }

            multiplyMatrices<<<grid,block>>>(uMatrixDevice, sMatrixDevice,
              tempMatrixDevice, numPixels, numSingularValues, numSingularValues);

            CudaCheckError();

            CudaSafeCall(cudaFree(sMatrixDevice));
            CudaSafeCall(cudaFree(uMatrixDevice));

            float* vtMatrixDevice;
            float* tempMatrix2Device;

            CudaSafeCall(cudaMalloc((void**)&vtMatrixDevice, numSingularValues * numTime
              * sizeof(float)));
            CudaSafeCall(cudaMalloc((void**)&tempMatrix2Device, numPixels
              * numTime * sizeof(float)));

            CudaSafeCall(cudaMemcpy(vtMatrixDevice, newVTMatrix, numSingularValues * numTime
              * sizeof(float), cudaMemcpyHostToDevice));

            a = numPixels;
            b = numTime;

            if(65535 > a*b){
              grid.x = a*b;
            }
            else if(65535*1024 > a*b){
              grid.x = 65535;
              block.x = 1024;
              while(block.x*grid.x > a*b){
                block.x--;
              }
            }
            else{
              grid.x = 65535;
              block.x = 1024;
              while(grid.x*grid.y*block.x < a*b){
                grid.y++;
              }
            }

            multiplyMatrices<<<grid,block>>>(tempMatrixDevice, vtMatrixDevice,
              tempMatrix2Device, numPixels, numSingularValues, numTime);

            CudaCheckError();

            CudaSafeCall(cudaFree(vtMatrixDevice));
            CudaSafeCall(cudaFree(tempMatrixDevice));

            float* originalMatrix = new float[numPixels*numTime];


            CudaSafeCall(cudaMemcpy(originalMatrix, tempMatrix2Device, numPixels*numTime*sizeof(float), cudaMemcpyDeviceToHost));
            CudaSafeCall(cudaFree(tempMatrix2Device));

            bool previousZero = false;
            int rowCounter = 0;

            for(int i = 0; i < numPixels*numTime; ++i){


              if(originalMatrix[i] == 0.0f) {

                //cout<< i <<endl;
                rowCounter++;

              }
              if (previousZero && rowCounter == 512) {

                std::cout << i/512<< " Row is all zero" << '\n';

              }
              if(originalMatrix[i] == 0.0f) {

                previousZero = true;

              }
              else {

                previousZero = false;

              }
              if (i%512 == 0) {

                rowCounter = 0;

              }
            }

            NMF(heightMatrix, widthMatrix, newUMatrix, newSMatrix, newVTMatrix, originalMatrix,
              numPixels, numTime, numSingularValues, targetLoss);

            cout << "NMF is done" << endl;

            ofstream heightMatrixFile("data/" + baseName + ".nmf_H");
            if (heightMatrixFile.is_open()) {

              for(long i = 0; i < (numSingularValues * numTime); i++){

                    heightMatrixFile << heightMatrix[i] << "\n";

               }

             }
             heightMatrixFile.close();
             cout<<"heightMatrix dumped"<<endl;

             ofstream widthMatrixFile("data/" + baseName + ".nmf_W");
             if (widthMatrixFile.is_open()) {

               for(long i = 0; i < (numPixels * numSingularValues); i++){

                     widthMatrixFile << widthMatrix[i] << "\n";

                }

              }
              widthMatrixFile.close();
              cout<<"widthMatrix dumped"<<endl;

          }
          else{
            cout<<"ERROR OPENING TIFF IN THIS DIRECTORY"<<endl;
            exit(-1);
          }

      }


      return 0;

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

void updateHeightMatrix(float* heightMatrix, float* widthMatrix,
  float* uMatrix, float* sMatrix, float* vtMatrix, float* newHeightMatrix,
  long numPixels, long numTime, long numSingularValues) {

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    long a = 1;
    long b = 1;

    float* widthMatrixTransposedDevice;
    float* uMatrixDevice;
    float* tempSquareMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&widthMatrixTransposedDevice, numPixels * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&uMatrixDevice, numPixels * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&tempSquareMatrixDevice, numSingularValues
      * numSingularValues * sizeof(float)));

    cout << "Starting to transpose matrix" << endl;

    float* widthMatrixTransposed = new float[numPixels * numSingularValues];

    cout << numPixels << endl;
    cout << numSingularValues << endl;

    for (long i = 0; i < numPixels; i++) {

      for (long j = 0; j < numSingularValues; j++) {

        //cout << "i = " << i << " j = " << j << " index = " << widthMatrix[i * numSingularValues + j] << endl;

        widthMatrixTransposed[j * numPixels + i] = widthMatrix[i * numSingularValues + j];

      }

    }

    cout << "Matrix transposed" << endl;

    CudaSafeCall(cudaMemcpy(widthMatrixTransposedDevice, widthMatrixTransposed, numPixels
      * numSingularValues * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(uMatrixDevice, uMatrix, numPixels * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    a = numSingularValues;
    b = numSingularValues;

    if(65535 > a*b){
      grid.x = a*b;
    }
    else if(65535*1024 > a*b){
      grid.x = 65535;
      block.x = 1024;
      while(block.x*grid.x > a*b){
        block.x--;
      }
    }
    else{
      grid.x = 65535;
      block.x = 1024;
      while(grid.x*grid.y*block.x < a*b){
        grid.y++;
      }
    }

    cout << "First multiplication kernel" << endl;

    multiplyMatrices<<<grid,block>>>(widthMatrixTransposedDevice, uMatrixDevice, tempSquareMatrixDevice,
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

    cout << "Second multiplication kernel" << endl;

    multiplyMatrices<<<grid,block>>>(tempSquareMatrixDevice, sMatrixDevice, tempSquareMatrix2Device,
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

    a = numSingularValues;
    b = numTime;

    if(65535 > a*b){
      grid.x = a*b;
    }
    else if(65535*1024 > a*b){
      grid.x = 65535;
      block.x = 1024;
      while(block.x*grid.x > a*b){
        block.x--;
      }
    }
    else{
      grid.x = 65535;
      block.x = 1024;
      while(grid.x*grid.y*block.x < a*b){
        grid.y++;
      }
    }

    cout << "Third multiplication kernel" << endl;

    multiplyMatrices<<<grid,block>>>(tempSquareMatrix2Device, vtMatrixDevice, numeratorDevice,
      numSingularValues, numSingularValues, numTime);

    CudaCheckError();

    CudaSafeCall(cudaFree(tempSquareMatrix2Device));
    CudaSafeCall(cudaFree(vtMatrixDevice));

    float* widthMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&widthMatrixDevice, numPixels
      * numSingularValues * sizeof(float)));

    CudaSafeCall(cudaMemcpy(widthMatrixDevice, widthMatrix, numPixels * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    a = numSingularValues;
    b = numSingularValues;

    if(65535 > a*b){
      grid.x = a*b;
    }
    else if(65535*1024 > a*b){
      grid.x = 65535;
      block.x = 1024;
      while(block.x*grid.x > a*b){
        block.x--;
      }
    }
    else{
      grid.x = 65535;
      block.x = 1024;
      while(grid.x*grid.y*block.x < a*b){
        grid.y++;
      }
    }

    cout << "Fourth multiplication kernel" << endl;

    multiplyMatrices<<<grid,block>>>(widthMatrixTransposedDevice, widthMatrixDevice, tempSquareMatrixDevice,
      numSingularValues, numPixels, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(widthMatrixTransposedDevice));
    CudaSafeCall(cudaFree(widthMatrixDevice));

    float* heightMatrixDevice;
    float* denominatorDevice;

    CudaSafeCall(cudaMalloc((void**)&heightMatrixDevice, numSingularValues
      * numTime * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&denominatorDevice, numSingularValues
      * numTime * sizeof(float)));

    CudaSafeCall(cudaMemcpy(heightMatrixDevice, heightMatrix, numSingularValues *
      numTime * sizeof(float), cudaMemcpyHostToDevice));

    a = numSingularValues;
    b = numTime;

    if(65535 > a*b){
      grid.x = a*b;
    }
    else if(65535*1024 > a*b){
      grid.x = 65535;
      block.x = 1024;
      while(block.x*grid.x > a*b){
        block.x--;
      }
    }
    else{
      grid.x = 65535;
      block.x = 1024;
      while(grid.x*grid.y*block.x < a*b){
        grid.y++;
      }
    }

    cout << "Fifth multiplication kernel" << endl;

    multiplyMatrices<<<grid,block>>>(tempSquareMatrixDevice, heightMatrixDevice,
      denominatorDevice, numSingularValues, numSingularValues, numTime);

    CudaCheckError();

    CudaSafeCall(cudaFree(tempSquareMatrixDevice));

    cout << "Scalar kernel" << endl;

    applyScalar<<<grid,block>>>(heightMatrixDevice, numeratorDevice,
      denominatorDevice, numSingularValues, numTime);

    CudaCheckError();

    CudaSafeCall(cudaMemcpy(newHeightMatrix, heightMatrixDevice, numSingularValues
      * numTime * sizeof(float), cudaMemcpyDeviceToHost));

    // for(int i = 0; i < numTime*numSingularValues; ++i){
    //   printf("%f became %f\n",heightMatrix[i],newHeightMatrix[i]);
    // }
    // exit(0);
    CudaSafeCall(cudaFree(heightMatrixDevice));
    CudaSafeCall(cudaFree(numeratorDevice));
    CudaSafeCall(cudaFree(denominatorDevice));

    delete[] widthMatrixTransposed;

  }
void updateWidthMatrix(float* heightMatrix, float* widthMatrix,
    float* uMatrix, float* sMatrix, float* vtMatrix, float* newWidthMatrix,
    long numPixels, long numTime, long numSingularValues) {

      dim3 grid = {1,1,1};
      dim3 block = {1,1,1};
      long a = 1;
      long b = 1;

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

      for (long i = 0; i < numSingularValues; i++) {

        for (long j = 0; j < numTime; j++) {

          heightMatrixTransposed[j * numSingularValues + i] = heightMatrix[i * numTime + j];

        }

      }

      CudaSafeCall(cudaMemcpy(heightMatrixTransposedDevice, heightMatrixTransposed,
        numSingularValues * numTime * sizeof(float), cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(vtMatrixDevice, vtMatrix, numSingularValues * numTime
        * sizeof(float), cudaMemcpyHostToDevice));

      a = numSingularValues;
      b = numSingularValues;

      if(65535 > a*b){
        grid.x = a*b;
      }
      else if(65535*1024 > a*b){
        grid.x = 65535;
        block.x = 1024;
        while(block.x*grid.x > a*b){
          block.x--;
        }
      }
      else{
        grid.x = 65535;
        block.x = 1024;
        while(grid.x*grid.y*block.x < a*b){
          grid.y++;
        }
      }

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
        tempSquareMatrix2Device, numSingularValues, numSingularValues, numSingularValues);

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

      a = numPixels;
      b = numSingularValues;

      if(65535 > a*b){
        grid.x = a*b;
      }
      else if(65535*1024 > a*b){
        grid.x = 65535;
        block.x = 1024;
        while(block.x*grid.x > a*b){
          block.x--;
        }
      }
      else{
        grid.x = 65535;
        block.x = 1024;
        while(grid.x*grid.y*block.x < a*b){
          grid.y++;
        }
      }

      multiplyMatrices<<<grid,block>>>(uMatrixDevice, tempSquareMatrix2Device,
        numeratorDevice, numPixels, numSingularValues, numSingularValues);

      CudaCheckError();

      CudaSafeCall(cudaFree(uMatrixDevice));
      CudaSafeCall(cudaFree(tempSquareMatrix2Device));

      float* heightMatrixDevice;

      CudaSafeCall(cudaMalloc((void**)&heightMatrixDevice, numSingularValues * numTime
        * sizeof(float)));

      CudaSafeCall(cudaMemcpy(heightMatrixDevice, heightMatrix, numSingularValues
        * numTime * sizeof(float), cudaMemcpyHostToDevice));

      a = numSingularValues;
      b = numSingularValues;

      if(65535 > a*b){
        grid.x = a*b;
      }
      else if(65535*1024 > a*b){
        grid.x = 65535;
        block.x = 1024;
        while(block.x*grid.x > a*b){
          block.x--;
        }
      }
      else{
        grid.x = 65535;
        block.x = 1024;
        while(grid.x*grid.y*block.x < a*b){
          grid.y++;
        }
      }

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

       a = numPixels;
       b = numSingularValues;

       if(65535 > a*b){
         grid.x = a*b;
       }
       else if(65535*1024 > a*b){
         grid.x = 65535;
         block.x = 1024;
         while(block.x*grid.x > a*b){
           block.x--;
         }
       }
       else{
         grid.x = 65535;
         block.x = 1024;
         while(grid.x*grid.y*block.x < a*b){
           grid.y++;
         }
       }

       multiplyMatrices<<<grid,block>>>(widthMatrixDevice, tempSquareMatrixDevice,
         denominatorDevice, numPixels, numSingularValues, numSingularValues);

       CudaCheckError();

       CudaSafeCall(cudaFree(tempSquareMatrixDevice));

       applyScalar<<<grid,block>>>(widthMatrixDevice, numeratorDevice,
         denominatorDevice, numPixels, numSingularValues);

       CudaCheckError();

       CudaSafeCall(cudaMemcpy(newWidthMatrix, widthMatrixDevice, numPixels
         * numSingularValues * sizeof(float), cudaMemcpyDeviceToHost));

       // for(int i = 0; i < numTime*numSingularValues; ++i){
       //
       //   printf("%f became %f\n",widthMatrix[i],newWidthMatrix[i]);
       // }
       // exit(0);
       CudaSafeCall(cudaFree(widthMatrixDevice));
       CudaSafeCall(cudaFree(numeratorDevice));
       CudaSafeCall(cudaFree(denominatorDevice));

       delete[] heightMatrixTransposed;

}

void NMF(float* heightMatrix, float* widthMatrix, float* uMatrix,
  float* sMatrix, float* vtMatrix, float* originalMatrix, long numPixels, long numTime,
  long numSingularValues, float targetLoss) {

    float* newWidthMatrix = new float[numPixels * numSingularValues];
    float* newHeightMatrix = new float[numSingularValues * numTime];

    cout << "New versions allocated" << endl;

    float loss = numeric_limits<float>::max();

    while(loss > targetLoss) {

      cout << "Updating Height Matrix" << endl;

      updateHeightMatrix(heightMatrix, widthMatrix, uMatrix, sMatrix, vtMatrix,
        newHeightMatrix, numPixels, numTime, numSingularValues);

      cout << "Updating Width Matrix" << endl;

      updateWidthMatrix(heightMatrix, widthMatrix, uMatrix, sMatrix, vtMatrix,
        newWidthMatrix, numPixels, numTime, numSingularValues);

      float* newWidthMatrixDevice;
      float* newHeightMatrixDevice;
      float* testMatrixDevice;

      CudaSafeCall(cudaMalloc((void**)&newWidthMatrixDevice, numPixels * numSingularValues
        * sizeof(float)));
      CudaSafeCall(cudaMalloc((void**)&newHeightMatrixDevice, numSingularValues * numTime
        * sizeof(float)));
      CudaSafeCall(cudaMalloc((void**)&testMatrixDevice, numPixels * numTime
        * sizeof(float)));

      CudaSafeCall(cudaMemcpy(newWidthMatrixDevice, newWidthMatrix, numPixels
        * numSingularValues * sizeof(float), cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(newHeightMatrixDevice, newHeightMatrix, numSingularValues
        * numTime * sizeof(float), cudaMemcpyHostToDevice));

      dim3 grid = {1,1,1};
      dim3 block = {1,1,1};

      long a = numPixels;
      long b = numTime;

      if(65535 > a*b){
        grid.x = a*b;
      }
      else if(65535*1024 > a*b){
        grid.x = 65535;
        block.x = 1024;
        while(block.x*grid.x > a*b){
          block.x--;
        }
      }
      else{
        grid.x = 65535;
        block.x = 1024;
        while(grid.x*grid.y*block.x < a*b){
          grid.y++;
        }
      }

      multiplyMatrices<<<grid,block>>>(newWidthMatrixDevice, newHeightMatrixDevice, testMatrixDevice,
        numPixels, numSingularValues, numTime);

      CudaCheckError();
      CudaSafeCall(cudaMemcpy(heightMatrix, newHeightMatrixDevice, numTime*numSingularValues*sizeof(float), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemcpy(widthMatrix, newWidthMatrixDevice, numPixels*numSingularValues*sizeof(float), cudaMemcpyDeviceToHost));


      CudaSafeCall(cudaFree(newWidthMatrixDevice));
      CudaSafeCall(cudaFree(newHeightMatrixDevice));
      float* originalMatrixDevice;

      CudaSafeCall(cudaMalloc((void**)&originalMatrixDevice, numPixels * numTime
        * sizeof(float)));

      CudaSafeCall(cudaMemcpy(originalMatrixDevice, originalMatrix, numPixels
        * numTime * sizeof(float), cudaMemcpyHostToDevice));

      grid = {50,1,1};
      block = {192,1,1};

      float* calculatedLoss;
      float temp = 0.0f;
      CudaSafeCall(cudaMalloc((void**)&calculatedLoss,sizeof(float)));
      CudaSafeCall(cudaMemcpy(calculatedLoss, &temp, sizeof(float), cudaMemcpyHostToDevice));;

      calculateLoss<<<grid,block>>>(originalMatrixDevice, testMatrixDevice, numPixels,
        numTime, calculatedLoss);
      CudaCheckError();

      CudaSafeCall(cudaMemcpy(&loss, calculatedLoss, sizeof(float), cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(originalMatrixDevice));
      CudaSafeCall(cudaFree(testMatrixDevice));

      cout << "Current Loss = " << loss << endl;

    }

  }

float findA(float* uMatrix, float* sMatrix, float* vtMatrix,
  long numPixels, long numTime, long numSingularValues) {

    dim3 grid = {1,1,1};
    dim3 block = {1,1,1};
    long a = 1;
    long b = 1;

    float* uMatrixDevice;
    float* sMatrixDevice;
    float* tempMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&uMatrixDevice, numPixels * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&sMatrixDevice, numSingularValues * numSingularValues
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&tempMatrixDevice, numPixels
      * numSingularValues * sizeof(float)));

    CudaSafeCall(cudaMemcpy(uMatrixDevice, uMatrix, numPixels
      * numSingularValues * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(sMatrixDevice, sMatrix, numSingularValues * numSingularValues
      * sizeof(float), cudaMemcpyHostToDevice));

    a = numPixels;
    b = numSingularValues;

    if(65535 > a*b){
      grid.x = a*b;
    }
    else if(65535*1024 > a*b){
      grid.x = 65535;
      block.x = 1024;
      while(block.x*grid.x > a*b){
        block.x--;
      }
    }
    else{
      grid.x = 65535;
      block.x = 1024;
      while(grid.x*grid.y*block.x < a*b){
        grid.y++;
      }
    }

    multiplyMatrices<<<grid,block>>>(uMatrixDevice, sMatrixDevice, tempMatrixDevice,
      numPixels, numSingularValues, numSingularValues);

    CudaCheckError();

    CudaSafeCall(cudaFree(uMatrixDevice));
    CudaSafeCall(cudaFree(sMatrixDevice));

    float* vtMatrixDevice;
    float* finalMatrixDevice;

    CudaSafeCall(cudaMalloc((void**)&vtMatrixDevice, numSingularValues * numTime
      * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&finalMatrixDevice, numPixels * numTime
      * sizeof(float)));

    CudaSafeCall(cudaMemcpy(vtMatrixDevice, vtMatrix, numSingularValues
      * numTime * sizeof(float), cudaMemcpyHostToDevice));

    a = numPixels;
    b = numTime;

    if(65535 > a*b){
      grid.x = a*b;
    }
    else if(65535*1024 > a*b){
      grid.x = 65535;
      block.x = 1024;
      while(block.x*grid.x > a*b){
        block.x--;
      }
    }
    else{
      grid.x = 65535;
      block.x = 1024;
      while(grid.x*grid.y*block.x < a*b){
        grid.y++;
      }
    }

    multiplyMatrices<<<grid,block>>>(tempMatrixDevice, vtMatrixDevice, finalMatrixDevice,
      numPixels, numSingularValues, numTime);

    CudaCheckError();

    CudaSafeCall(cudaFree(vtMatrixDevice));
    CudaSafeCall(cudaFree(tempMatrixDevice));

    float* finalMatrix = new float[numPixels*numTime];

    CudaSafeCall(cudaMemcpy(finalMatrix, finalMatrixDevice, numPixels
      * numTime * sizeof(float), cudaMemcpyDeviceToHost));

    float lowestValue = 0.0;

    for (long i = 0; i < (numPixels * numTime); i++) {

      if (finalMatrix[i] < lowestValue) {

        lowestValue = finalMatrix[i];

      }

    }

    if (lowestValue < 0.0) {

      lowestValue = lowestValue * (-1.0);
      lowestValue = lowestValue + 1;
      std::cout << lowestValue << '\n';

      return lowestValue;

    }
    else {

      lowestValue = lowestValue * (-1.0);
      lowestValue = lowestValue - 1;
      std::cout << lowestValue << '\n';

      return lowestValue;

    }

}


__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, long diffDimA,
   long comDim, long diffDimB) {

     long blockID = blockIdx.y * gridDim.x + blockIdx.x;
     long globalID = blockID * blockDim.x + threadIdx.x;
     long currentIndex = globalID;

     if (currentIndex < (diffDimA * diffDimB)) {

       long iIndex = currentIndex / diffDimB;
       long jIndex = currentIndex % diffDimB;

       float sum = 0;

       for (int k = 0; k < comDim; k++) {

         sum += (matrixA[iIndex * comDim + k] * matrixB[k * diffDimB + jIndex]);

       }

       matrixC[iIndex * diffDimB + jIndex] = sum;

     }

   }
__global__ void applyScalar(float* targetMatrix, float* numerator, float* denominator,
    long numRows, long numCols) {

    long blockID = blockIdx.y * gridDim.x + blockIdx.x;
    long globalID = blockID * blockDim.x + threadIdx.x;
    long currentIndex = globalID;
    long numThreads =  blockDim.x*gridDim.x*gridDim.y;

    while(currentIndex < (numCols * numRows)){

      targetMatrix[currentIndex] = targetMatrix[currentIndex]
        * (numerator[currentIndex] / denominator[currentIndex]);

      //printf("%f,%f\n",numerator[currentIndex], denominator[currentIndex]);

        currentIndex += numThreads;
    }
  }
__global__ void calculateLoss(float* originalMatrix, float* newMatrix, long numRows, long numCols, float* loss) {

  long blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  long currentIndex = globalID;
  long numThreads = blockDim.x * gridDim.x * gridDim.y;
  float localLoss = 0.0f;
  __shared__ float blockLoss;
  blockLoss = 0;
  __syncthreads();
  while (currentIndex  < (numRows * numCols)) {
    localLoss += abs(originalMatrix[currentIndex] - newMatrix[currentIndex]);
    //if(threadIdx.x == 0) printf("%d , %f - %f\n",currentIndex,originalMatrix[currentIndex], newMatrix[currentIndex]);

    currentIndex += numThreads;
  }
  atomicAdd(&blockLoss, localLoss);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(loss, blockLoss);
  }
}
