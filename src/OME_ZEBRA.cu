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
vector<uint32*> extractMartrices(TIFF* tif, string fileName);
vector<uint32*> extractMartrices(TIFF* tif);
vector<uint32> flattenMatrix(vector<uint32*> matrix, int cols, int rows);
uint32** hostTranspose(uint32** matrix, int rows, int cols);
__global__ void transposeuint32Matrix(uint32* flatOrigin, uint32* flatTransposed, long Nrows, long Ncols);
uint32 findMin(uint32* flatMatrix, int size);
__global__ void calcCa(uint32* flatMatrix, uint32 min);


int main(int argc, char *argv[]) {

    vector<vector<uint32*>> fullTiffVector;
    if(argc!=2) {
      cout << "Usage: ./exe <file>";
      return 1;
    }
    else {
      TIFF* tif = TIFFOpen(argv[1], "r");
      cout<<endl<<argv[1]<<" IS OPENED\n"<<endl;
      string fileName = argv[1];
      int dircount = 0;
      if (tif) {
          do {
            vector<uint32*> matrix;
            if(dircount == 0) matrix = extractMartrices(tif, fileName);
            else matrix = extractMartrices(tif);
            fullTiffVector.push_back(matrix);

            dircount++;
          }
          while (TIFFReadDirectory(tif));
          printf("%d directories in %s\n", dircount, argv[1]);

          //to save numColumns
          uint32 numColumns;
          TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &numColumns);

          TIFFClose(tif);

          //access matrix with fullTiffVector = timePoints
          printf("Total TimePoints = %d\nTotal Rows = %d\nTotal Columns = %d\n",fullTiffVector.size(), fullTiffVector[0].size(), numColumns);

          //structure of final vector is:
          //timePoints = vector<rows>
          //  rows = vector<columns>
          //    columns = uint32[numColumns]
          vector<vector<uint32>> flattenedTimePoints;

          //prepare arrays
          //flatten time points
          cout << "Preparing to flatten" << endl;
          for(int i = 0; i < fullTiffVector.size(); ++i){
            flattenedTimePoints.push_back(flattenMatrix(fullTiffVector[i],numColumns, fullTiffVector[i].size()));
          }

          //long numPixels = ((int)fullTiffVector[0].size())*numColumns;
          //bool* key = new bool[numPixels];
          //for(int i = 0; i < numPixels; +i) key[i] = false;

          //need to do calcCa

          cout << "Preparing to convert to array and transpose" << endl;

          int NNormal = 512;
          int MNormal = (1024*512);

          bool* key = new bool[MNormal];

          for (int i = 0; i < MNormal; i++) {
            key[i] = false;
          }

          uint32* temp;
          temp = new uint32[MNormal*NNormal];
          int indexOfTemp = 0;
          int nonZeroCounter = 0;
          uint32* rowArray = new uint32[512];
          int rowArrayIndex = 0;
          int lastGoodIndex = 0;

          for(unsigned i=0; (i < MNormal); i++) {
            nonZeroCounter = 0;
            rowArrayIndex = 0;
            for(unsigned j=0; (j < NNormal); j++) {
              if (flattenedTimePoints[j][i] != 0){
                nonZeroCounter++;
              }
              rowArray[rowArrayIndex] = flattenedTimePoints[j][i];
              rowArrayIndex++;
            }

            if (nonZeroCounter != 0) {
              for (int k = 0; k < 512; k++) {
                temp[indexOfTemp] = rowArray[k];
                indexOfTemp++;
              }
              lastGoodIndex++;
              key[i] = true;
            }

          }
          cout << lastGoodIndex << endl;
          uint32* actualArray = new uint32[(lastGoodIndex)*512];
          cout << "test1" << endl;
          for (long i = 0; i < ((lastGoodIndex) * 512); i++) {

            actualArray[i] = temp[i] + 1;

          }

          cout << "Dumping to File" << endl;

          ofstream myfile ("data/NNMF.csv");
          if (myfile.is_open()) {
            for(long count = 0; count < ((lastGoodIndex) * 512); count++){

              if ((count + 1) % 512 == 0) {

                 myfile << actualArray[count] << "\n" ;

              }
              else {

                myfile << actualArray[count] << " " ;

              }
            }
            myfile.close();
          }

          ofstream mykeyfile ("data/key.csv");
          if (mykeyfile.is_open()) {
            for(long i = 0; i < MNormal; i++){

               mykeyfile << key[i] << "\n" ;

             }

           }
            mykeyfile.close();
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

vector<uint32*> extractMartrices(TIFF* tif, string fileName){
  string newtiff = fileName.substr(0, fileName.length() - 8) + "_TP1.tif";
  TIFF* firstTimePoint = TIFFOpen(newtiff.c_str(), "w");
  if(firstTimePoint){
    tdata_t buf;
    uint32 row;
    uint32 config;
    vector<uint32*> currentPlane;

    uint32 height, width, samplesPerPixel, bitsPerSample, photo, scanLineSize;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
    TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &scanLineSize);


    TIFFSetField(firstTimePoint, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(firstTimePoint, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(firstTimePoint, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
    TIFFSetField(firstTimePoint, TIFFTAG_BITSPERSAMPLE,bitsPerSample);
    TIFFSetField(firstTimePoint, TIFFTAG_PHOTOMETRIC, photo);
    TIFFSetField(firstTimePoint, TIFFTAG_ROWSPERSTRIP, scanLineSize);
    cout<<"\nTIMEPOINT 1 .tif info:"<<endl;
    printf("width = %d\nheight = %d\nsamplesPerPixel = %d\nbitsPerSample = %lo\nscanLineSize = %d\n\n",width,height,samplesPerPixel,bitsPerSample,photo,scanLineSize);
    buf = _TIFFmalloc(TIFFScanlineSize(tif));
    uint32* data;
    ofstream test("data/TP1.csv");
    //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
    for (row = 0; row < height; row++){
      TIFFReadScanline(tif, buf, row);
      data=(uint32*)buf;
      for(int i = 0; i < width; ++i){
        test<<data[i];
        if(i != width - 1) test<<",";
      }
      test<<"\n"<<endl;
      if(TIFFWriteScanline(firstTimePoint, buf, row) != 1){
        cout<<"ERROR WRITING FIRST TIMEPOINT"<<endl;
        exit(-1);
      }

      currentPlane.push_back(data);
    }
    test.close();
    TIFFClose(firstTimePoint);
    _TIFFfree(buf);
    return currentPlane;
  }
  else{
    cout<<"COULD NOT CREATE FIRST TIMEPOINT TIFF"<<endl;
    exit(-1);
  }
}
vector<uint32*> extractMartrices(TIFF* tif){

  uint32 height,width;
  tdata_t buf;
  uint32 row;
  uint32 config;
  vector<uint32*> currentPlane;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  buf = _TIFFmalloc(TIFFScanlineSize(tif));

  uint32* data;
  //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
  for (row = 0; row < height; row++){
    TIFFReadScanline(tif, buf, row);
    data=(uint32*)buf;
    currentPlane.push_back(data);

    //printArray(data,width);//make sure you have a big screen
  }
  //cout<<endl<<endl;//if you are using the printArray method
  _TIFFfree(buf);
  return currentPlane;
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
__global__ void calcCa(uint32* flatMatrix, uint32 min){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  flatMatrix[globalID] = flatMatrix[globalID] - min;
}
