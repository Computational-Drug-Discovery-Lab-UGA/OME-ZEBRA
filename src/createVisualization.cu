/*
Created and edited by Jackson Parker and Godfrey Hendrix
This script is used to create a visualized tif image of the output
of NNMF run on a tif directory. The necessary inputs include a tif image
of the first timepoint, the w matrix of NNMF output and a  key csv that
represents the if a pixel row is all 0.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "tiffio.h"
#include <cstring>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits>



using namespace std;


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


string createFourCharInt(int i);
__global__ void multiplyMatrices(float* matrixA, float* matrixB, float* matrixC, long diffDimA,
   long comDim, long diffDimB);

int main(int argc, char *argv[]) {

  if (argc < 2) {
    cout << "Usage: ./exe <file>";
    return 1;
  } else {
    int k = 2;
    string baseDirectoryIn = "/media/spacey-person/CDDL_Drive/Registered/";
    string baseDirectoryOut = "/media/spacey-person/CDDL_Drive/NNMF_NOSVD/";
    string wFileLocation;
    string hFileLocation;
    string tifName = argv[1];
    string tifFile = baseDirectoryIn + tifName + "/" + tifName + "0000.tif";
    string keyFileLocation = baseDirectoryOut + tifName + "/key.csv";
    TIFF *tif = TIFFOpen(tifFile.c_str(), "r");
    wFileLocation = baseDirectoryOut + tifName + "/NNMF.nmf_W.txt";
    hFileLocation = baseDirectoryOut + tifName + "/NNMF.nmf_H.txt";
    if (argc == 3) {
      istringstream argK(argv[2]);
      argK >> k;
    }
    cout << "k = " << k << endl;

    if (tif) {

      tdata_t buf;
      tsize_t scanLineSize;
      uint32 row;
      vector<uint32 *> currentPlane;

      uint32 height, width, samplesPerPixel, bitsPerSample, photo;

      TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
      TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
      TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
      TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

      scanLineSize = TIFFScanlineSize(tif);
      buf = _TIFFmalloc(scanLineSize);

      uint32 max = 0;
      uint32 min = 4294967295;
      uint32 ***kMatrix = new uint32**[k];
      for (int i = 0; i < k; ++i) {
        kMatrix[i] = new uint32*[height];
        for (int ii = 0; ii < height; ++ii) {
          kMatrix[i][ii] = new uint32[width];
        }
      }

      for (row = 0; row < height; ++row) {
        if (TIFFReadScanline(tif, buf, row, 0) != -1) {
          for(int i = 0; i < k; ++i){
            memcpy(kMatrix[i][row], buf, scanLineSize);
          }
          for (int col = 0; col < width; ++col) {
            if (kMatrix[0][row][col] > max)
              max = kMatrix[0][row][col];
            if (kMatrix[0][row][col] < min)
              min = kMatrix[0][row][col];
          }
        } else {
          cout << "ERROR READING SCANLINE" << endl;
          exit(-1);
        }
      }
      cout<<"min = "<<min<<" max = "<<max<<endl;
      cout<<"done reading in tp 0000"<<endl;
      TIFFClose(tif);
      _TIFFfree(buf);
      string wLine;
      string hLine;
      string keyLine;
      wLine = "";
      ifstream wFile(wFileLocation);
      ifstream keyFile(keyFileLocation);
      ifstream hFile(hFileLocation);
      vector<vector<float>> wMatrix;
      vector<vector<float>> hMatrix;
      vector<bool> keyVector;
      vector<float> blankVector;
      for (int i = 0; i < k; ++i) {
        blankVector.push_back(0.0f);
      }

      if (wFile.is_open() && keyFile.is_open()) {

        int largest = 0;
        float largestValue = 0.0f;
        float currentValue = 0.0f;

        for (row = 0; row < height; ++row) {
          keyLine = "";
          getline(keyFile, keyLine);
          if (keyLine == "1") {
            keyVector.push_back(true);
            for (int col = 0; col < width; ++col) {
              wLine = "";
              getline(wFile, wLine);
              istringstream ss(wLine);
              vector<float> currentVector;
              largest = 0;
              largestValue = 0.0f;
              currentValue = 0.0f;
              for (int kFocus = 0; kFocus < k; kFocus++) {
                kMatrix[kFocus][row][col] -= min;
                ss >> currentValue;

                if (largestValue < currentValue) {
                  largest = kFocus;
                  largestValue = currentValue;
                }
                currentVector.push_back(currentValue);
              }
              kMatrix[largest][row][col] += (max - min) / 2;
              wMatrix.push_back(currentVector);
            }
          } else {
            keyVector.push_back(false);
            wMatrix.push_back(blankVector);
          }
        }
        cout<<"done creating spacial matrix[k][row][col]"<<endl;
        wFile.close();
        keyFile.close();
      } else {
        cout << "FAILURE OPENING W_NMF file or key file" << endl;
        exit(-1);
      }
      cout<<"starting k spacial image generation"<<endl;
      for (int kFocus = 0; kFocus < k; ++kFocus) {
        string fileName = baseDirectoryOut + tifName + "/" +
        tifName + "_" + to_string(k) + "_" + to_string(kFocus) + ".tif";
        TIFF *resultTif = TIFFOpen(fileName.c_str(), "w");
        if (resultTif) {
          TIFFSetField(resultTif, TIFFTAG_IMAGEWIDTH, width);
          TIFFSetField(resultTif, TIFFTAG_IMAGELENGTH, height);
          TIFFSetField(resultTif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
          TIFFSetField(resultTif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
          TIFFSetField(resultTif, TIFFTAG_PHOTOMETRIC, photo);
          for(int row = 0; row < height; ++row){
            if (TIFFWriteScanline(resultTif, kMatrix[kFocus][row], row, 0) != 1) {
              cout << "ERROR WRITING FIRST TIMEPOINT" << endl;
              exit(-1);
            }
          }
          cout<<fileName<<" has been created"<<endl;
          TIFFClose(resultTif);
        }
        else{
          cout<<"COULD NOT OPEN TIF"<<endl;
          exit(-1);
        }
      }
      for (int i = 0; i < k; ++i) {
        for (int ii = 0; ii < height; ++ii) {
          delete[] kMatrix[i][ii];
        }
        delete[] kMatrix[i];
      }
      delete[] kMatrix;
      int numTimePoints = 0;

      if (hFile.is_open()) {
        float currentValue = 0.0f;
        for (int i = 0; i < k; ++i) {
          getline(hFile, hLine);
          vector<float> currentVector;
          istringstream ss(hLine);
          while (ss >> currentValue) {
            currentVector.push_back(currentValue);
            ++numTimePoints;
          }
          hMatrix.push_back(currentVector);
          hLine = "";
        }
        hFile.close();
        numTimePoints /= k;
      } else {
        cout << "FAILURE OPENING H_NMF" << endl;
        exit(-1);
      }
      cout<<"done reading h matrix"<<endl;
      float* wColDevice;
      float* hRowDevice;
      float* resultDevice;
      float* result = new float[height*width*numTimePoints];
      float* wCol = new float[height*width];
      float* hRow = new float[numTimePoints];
      CudaSafeCall(cudaMalloc((void**)&wColDevice, height*width*sizeof(float)));
      CudaSafeCall(cudaMalloc((void**)&hRowDevice, numTimePoints*sizeof(float)));
      CudaSafeCall(cudaMalloc((void**)&resultDevice, width*height*numTimePoints*sizeof(float)));
      dim3 grid = {1,1,1};
      dim3 block = {1,1,1};
      int a = height*width;
      int b = numTimePoints;

      if(65535 > a*b){
        grid.x = a*b;
      }
      else if(65535*1024 > a*b){
        grid.x = 65535;
        block.x = 1024;
        while(block.x*grid.x > a*b){
          block.x--;
        }
        block.x++;
      }
      else{
        grid.x = 65535;
        block.x = 1024;
        while(grid.x*grid.y*block.x < a*b){
          grid.y++;
        }
      }
      uint32 **data = new uint32*[height];
      for(int i = 0; i < height; ++i){
        data[i] = new uint32[width];
      }
      cout<<"starting k video generation"<<endl;
      for(int kFocus = 0; kFocus < k; ++kFocus){
        string newDirectoryName = baseDirectoryOut + tifName + "/" + tifName + "_k" + to_string(k) + "_" + to_string(kFocus);
        if(mkdir(newDirectoryName.c_str(), 0777) == -1){
          cout<<"CANNOT CREATE "<<newDirectoryName<<endl;
          exit(-1);
        }
        cout<<newDirectoryName<<" created"<<endl;
        for(int w = 0; w < height*width; ++w){
          wCol[w] = wMatrix[w][kFocus];
        }
        for(int h = 0; h < numTimePoints; ++h){
          hRow[h] = hMatrix[kFocus][h];
        }
        CudaSafeCall(cudaMemcpy(wColDevice, wCol, height*width*sizeof(float), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(hRowDevice, hRow, numTimePoints*sizeof(float), cudaMemcpyHostToDevice));
        multiplyMatrices<<<grid,block>>>(wColDevice, hRowDevice, resultDevice, height*width, 1, numTimePoints);
        CudaCheckError();
        CudaSafeCall(cudaMemcpy(result, resultDevice, width*height*numTimePoints*sizeof(float), cudaMemcpyDeviceToHost));
        float minF = numeric_limits<float>::max();
        float maxF = numeric_limits<float>::min();
        for(int i = 0; i < height*width*numTimePoints; ++i){
          if(result[i] < minF) minF = result[i];
          if(result[i] > maxF) maxF = result[i];
        }
        for(int i = 0; i < height*width*numTimePoints; ++i){
          result[i] = 4294967295*(result[i] - minF)/(maxF-minF);
        }
        for(int tp = 0; tp < numTimePoints; ++tp){
          string newTif = newDirectoryName + "/" + tifName + "_" + createFourCharInt(tp);
          TIFF *tpfTif = TIFFOpen(newTif.c_str(), "w");

          if(tpfTif){
            TIFFSetField(tpfTif, TIFFTAG_IMAGEWIDTH, width);
            TIFFSetField(tpfTif, TIFFTAG_IMAGELENGTH, height);
            TIFFSetField(tpfTif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
            TIFFSetField(tpfTif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
            TIFFSetField(tpfTif, TIFFTAG_PHOTOMETRIC, photo);
            for(int row = 0; row < height; ++row){
              for(int col = 0; col < width; ++col){
                data[row][col] = (uint32) result[(row*width + col)*numTimePoints + tp];
              }

              if (TIFFWriteScanline(tpfTif, data[row], row, 0) != 1) {
                cout << "ERROR WRITING FIRST TIMEPOINT" << endl;
                exit(-1);
              }
            }
            TIFFClose(tpfTif);
            cout<<newTif<<" has been created"<<endl;
          }
          else{
            cout<<"COULD NOT CREATE "<<newTif<<endl;
            exit(-1);
          }
        }
      }
      for(int i = 0; i < height; ++i){
        delete[] data[i];
      }
      delete[] data;
    } else {
      cout << "COULD NOT OPEN " << argv[1] << endl;
      exit(-1);
    }
  }
  return 0;
}

string createFourCharInt(int i) {
  string strInt;
  if (i < 10) {
    strInt = "000" + to_string(i);
  } else if (i < 100) {
    strInt = "00" + to_string(i);
  } else if (i < 1000) {
    strInt = "0" + to_string(i);
  } else {
    strInt = to_string(i);
  }
  return strInt;
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
