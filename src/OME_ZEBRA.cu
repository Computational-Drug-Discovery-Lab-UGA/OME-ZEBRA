#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include "tiffio.h"
#include <fstream>

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
void printArray(uint16 * array, uint16 width);
vector<uint16*> extractMartrices(TIFF* tif, string fileName);
vector<uint16*> extractMartrices(TIFF* tif);
vector<uint16> flattenMatrix(vector<uint16*> matrix, int cols, int rows);
uint16** hostTranspose(uint16** matrix, int rows, int cols);
__global__ void transposeuint16Matrix(uint16* flatOrigin, uint16* flatTransposed, long Nrows, long Ncols);
uint16 findMin(uint16* flatMatrix, int size);
__global__ void calcCa(uint16* flatMatrix, uint16 min);


int main(int argc, char *argv[]) {

    vector<vector<uint16*>> fullTiffVector;
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
            vector<uint16*> matrix;
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
          //    columns = uint16[numColumns]
          vector<vector<uint16>> flattenedTimePoints;

          //prepare arrays
          //flatten time points
          cout << "Preparing to flatten" << endl;
          for(int i = 0; i < fullTiffVector.size(); ++i){
            flattenedTimePoints.push_back(flattenMatrix(fullTiffVector[i],numColumns, fullTiffVector[i].size()));
          }
          //transpose time TimePoints

          cout << "Preparing to convert to array and transpose" << endl;
          int NNormal = 512;
          int MNormal = (1024*512);

          bool* key = new bool[MNormal];

          for (int i = 0; i < MNormal; i++) {
            key[i] = false;
          }

          uint16* temp;
          temp = new uint16[MNormal*NNormal];
          int indexOfTemp = 0;
          int nonZeroCounter = 0;
          uint16* rowArray = new uint16[512];
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
          uint16* actualArray = new uint16[(lastGoodIndex)*512];
          cout << "test1" << endl;
          for (long i = 0; i < ((lastGoodIndex) * 512); i++) {
            actualArray[i] = temp[i];
          }

           cout << "Dumping to File" << endl;

           ofstream myfile("data/new.csv");
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

           // ifstream inFile ("A.csv");
           //  long count = (512*2048*1024);
           //  uint16* testArray;
           //  testArray = new uint16[count];
           //  printf("Test\n");
           //  for(long i = 0; i < count; i++){
           //    inFile >> testArray[i];
           //  }
           //  inFile.close();
           //
           // for (long i = 0; i < count; i++) {
           //   if (temp[i] != testArray[i]) {
           //     cout << "Not equal" << endl;
           //   }
           //   if (testArray[i] != 0) {
           //     cout << testArray[i] << endl;
           //   }
           // }


          //SVD


          //nnmf



      else{
        cout<<"COULD NOT OPEN"<<argv[1];
        return 1;
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
void printArray(uint16 * array, uint16 width){
    uint32 i;
    for (i=0;i<width;i++){
      printf("%u ", array[i]);
    }
    cout<<endl;
}

vector<uint16*> extractMartrices(TIFF* tif, string fileName){

  uint32 height,width;
  tdata_t buf;
  uint32 row;
  uint32 config;
  vector<uint16*> currentPlane;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  buf = _TIFFmalloc(TIFFScanlineSize(tif));

  uint16* data;
  string newfile = fileName.substr(0, fileName.length() - 8) + "_TP1.csv";
  ofstream myfile(newfile);

  //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
  for (row = 0; row < height; row++){
    TIFFReadScanline(tif, buf, row);
    data=(uint16*)buf;
    currentPlane.push_back(data);
    for(int i = 0; i < width; ++i){
      if(i != width -1){
        if(!myfile.is_open()) exit(0);
        myfile<<data[i]<<",";
      }
      else{
        myfile<<data[i];
      }
    }
    myfile<<"\n";

    //printArray(data,width);//make sure you have a big screen
  }
  //cout<<endl<<endl;//if you are using the printArray method
  _TIFFfree(buf);
  return currentPlane;
}
vector<uint16*> extractMartrices(TIFF* tif){

  uint32 height,width;
  tdata_t buf;
  uint32 row;
  uint32 config;
  vector<uint16*> currentPlane;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  buf = _TIFFmalloc(TIFFScanlineSize(tif));

  uint16* data;
  //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
  for (row = 0; row < height; row++){
    TIFFReadScanline(tif, buf, row);
    data=(uint16*)buf;
    currentPlane.push_back(data);

    //printArray(data,width);//make sure you have a big screen
  }
  //cout<<endl<<endl;//if you are using the printArray method
  _TIFFfree(buf);
  return currentPlane;
}

uint16** hostTranspose(uint16** matrix, int rows, int cols){
  uint16** transposable = new uint16*[rows];
  for(int row = 0; row < rows; ++row){
    transposable[row] = new uint16[cols];
    for(int col = 0; col < cols; ++col){
      transposable[row][col] = matrix[col][row];
    }
    //cout<<"Timepoint "<<row<<" trasposed..."<<endl;

  }

  return transposable;
}

__global__ void transposeuint16Matrix(uint16* flatOrigin, uint16* flatTransposed, long Nrows, long Ncols){

  long globalID = blockIdx.x * blockDim.x + threadIdx.x;
  long pixel = globalID;
  long stride = gridDim.x * blockDim.x;
  long flatLength = Nrows * Ncols;
  long row = 0;
  long col = 0;
  uint16 currentPixelIntensity = 0;
  while(pixel < flatLength){
    row = pixel/Ncols;
    col = pixel - Ncols*row;
    flatTransposed[pixel] = flatOrigin[row + Nrows*col];
    pixel += stride;
  }

}

vector<uint16> flattenMatrix(vector<uint16*> matrix, int cols, int rows){
  vector<uint16> flat;
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      flat.push_back(matrix[r][c]);
    }
  }
  //cout<<"Matrix is flattened."<<endl;
  return flat;
}
uint16 findMin(uint16* flatMatrix, int size){
  uint16 currentMin = 0;
  for(int i = 0; i < size; ++i){
    if(currentMin > flatMatrix[i]){
      currentMin = flatMatrix[i];
    }
  }
  return currentMin;
}
__global__ void calcCa(uint16* flatMatrix, uint16 min){
  int globalID = blockIdx.x * blockDim.x + threadIdx.x;
  flatMatrix[globalID] = flatMatrix[globalID] - min;
}
