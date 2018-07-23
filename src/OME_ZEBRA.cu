#include "common_includes.h"

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

/*
METHOD DECLARATIONS
*/

string createFourCharInt(int i);
uint32 *extractMartrices(TIFF *tif, string fileName);
uint32 *extractMartrices(TIFF *tif);
__global__ void normalize(uint32 *flatMatrix, float *normals, uint32 min,
                          uint32 max, long size);

/*
MAIN
*/

int main(int argc, char *argv[]) {

  if (argc != 3) {
    cout << "Usage: ./exe <file> <# of time points>";
    return 1;
  } else {
    string baseDirectoryIn = "/media/spacey-person/CDDL_Drive/Registered/";
    string baseDirectoryOut = "/media/spacey-person/CDDL_Drive/NNMF_NOSVD/";
    vector<uint32 *> flattenedTimePoints;
    string baseName = argv[1];
    int numTimePoints = atoi(argv[2]);
    if (numTimePoints == 0) {
      cout << "ERROR INVALID TIMEPOINTS" << endl;
      exit(-1);
    }
    bool allTifsAreGood = true;
    uint32 numColumns;
    uint32 numRows;
    string currentTif;
    dim3 grid = {1, 1, 1};
    dim3 block = {1, 1, 1};
    for (int i = 0; i < numTimePoints; ++i) {

      currentTif = baseDirectoryIn + baseName + "/" + baseName +
                   createFourCharInt(i) + ".tif";

      TIFF *tif = TIFFOpen(currentTif.c_str(), "r");

      if (tif) {
        if (i == 0) {
          TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &numColumns);
          TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &numRows);
        }
        uint32 tempCol;
        uint32 tempRow;
        cout << currentTif << " IS OPENED" << endl;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &tempCol);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &tempRow);
        if (numRows != tempRow || numColumns != tempCol) {
          cout << "ERROR NOT ALL TIFFS ARE THE SAME LENGTH" << endl;
          exit(-1);
        }

        uint32 *flatMatrix = new uint32[numRows * numColumns];
        flatMatrix = extractMartrices(tif);
        flattenedTimePoints.push_back(flatMatrix);
        TIFFClose(tif);

      } else {
        allTifsAreGood = false;
        break;
      }
    }
    if (allTifsAreGood) {

      int NNormal = numTimePoints;
      int MNormal = (numRows * numColumns);

      cout << "flattening" << endl;

      uint32 min = UINT32_MAX;
      uint32 max = 0;
      uint32 *temp = new uint32[MNormal * NNormal];
      int indexOfTemp = 0;
      int nonZeroCounter = 0;
      uint32 *rowArray = new uint32[numColumns];
      int rowArrayIndex = 0;
      for (unsigned i = 0; i < MNormal; i++) {

        nonZeroCounter = 0;
        rowArrayIndex = 0;
        for (unsigned j = 0; j < NNormal; j++) {
          if (flattenedTimePoints[j][i] != 0) {
            nonZeroCounter++;
            if (flattenedTimePoints[j][i] < min)
              min = flattenedTimePoints[j][i];
            if (flattenedTimePoints[j][i] > max)
              max = flattenedTimePoints[j][i];
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
      // need to delete all flattenedTimePoints arrays
      delete[] rowArray;

      uint32 *actualArray = new uint32[MNormal * NNormal];
      float *normalized = new float[MNormal * NNormal];
      cout << "loading arrays" << endl;

      for (long i = 0; i < MNormal * NNormal; i++) {
        actualArray[i] = temp[i];
      }

      if (65535 > MNormal * NNormal) {
        grid.x = MNormal * NNormal;
      } else if (65535 * 1024 > MNormal * NNormal) {
        grid.x = 65535;
        block.x = 1024;
        while (block.x * grid.x > MNormal * NNormal) {
          block.x--;
        }
        block.x++;
      } else {
        grid.x = 65535;
        block.x = 1024;
        while (grid.x * grid.y * block.x < MNormal * NNormal) {
          grid.y++;
        }
      }
      cout << "prepare for calcCa cuda kernel with min = " << min
           << ",max = " << max << endl;
      float *normalizedDevice;
      uint32 *actualArrayDevice;
      CudaSafeCall(cudaMalloc((void **)&actualArrayDevice,
                              MNormal * NNormal * sizeof(uint32)));
      CudaSafeCall(cudaMalloc((void **)&normalizedDevice,
                              MNormal * NNormal * sizeof(float)));
      CudaSafeCall(cudaMemcpy(actualArrayDevice, actualArray,
                              MNormal * NNormal * sizeof(uint32),
                              cudaMemcpyHostToDevice));
      normalize<<<grid, block>>>(actualArrayDevice, normalizedDevice, min, max,
                                 MNormal * NNormal);
      CudaCheckError();
      CudaSafeCall(cudaMemcpy(normalized, normalizedDevice,
                              MNormal * NNormal * sizeof(float),
                              cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaFree(actualArrayDevice));
      CudaSafeCall(cudaFree(normalizedDevice));
      delete[] actualArray;
      cout << "calcCa has completed applying offset" << endl;

      float *tempNorm = new float[MNormal * NNormal];
      indexOfTemp = 0;
      int lastGoodIndex = 0;
      cout << "Creating key" << endl;

      bool *key = new bool[MNormal];
      for (int i = 0; i < MNormal; i++) {
        key[i] = false;
      }
      for (unsigned i = 0; i < MNormal; i++) {
        nonZeroCounter = 0;
        for (unsigned j = 0; j < NNormal; j++) {
          if (normalized[(NNormal * i) + j] != 0.0f) {
            nonZeroCounter++;
            break;
          }
        }
        if (nonZeroCounter != 0) {
          for (int j = 0; j < NNormal; j++) {
            tempNorm[indexOfTemp] = normalized[(NNormal * i) + j];
            indexOfTemp++;
            key[i] = true;
          }
          lastGoodIndex++;
        } else {
          cout << "EMPTY ROW FOR PIXEL " << i << endl;
        }
      }
      cout << "NUMROWS = " << lastGoodIndex << endl;
      if (lastGoodIndex == NNormal - 1) {
        cout << "KEY CREATED BUT ALL PIXELS HAVE ATLEAST 1 NONZERO VALUE"
             << endl;
      }
      delete[] normalized;
      cout << "Dumping to File" << endl;

      ofstream myfile(baseDirectoryOut + baseName + "/NNMF.nmf");
      if (myfile.is_open()) {
        for (int i = 0; i < (lastGoodIndex)*NNormal; i++) {
          if ((i + 1) % 512 == 0) {
            myfile << tempNorm[i] << "\n";
          } else {
            myfile << tempNorm[i] << " ";
          }
        }
        myfile.close();
      }
      cout << "NNMF.nmf created successfuly" << endl;
      ofstream mykeyfile(baseDirectoryOut + baseName + "/key.csv");
      if (mykeyfile.is_open()) {
        for (long i = 0; i < MNormal; i++) {

          mykeyfile << key[i] << "\n";
        }
      }
      mykeyfile.close();
      cout << "key created succesfully" << endl;

    } else {
      cout << "ERROR OPENING TIFF IN THIS DIRECTORY" << endl;
      exit(-1);
    }
  }
  return 0;
}

/*
METHOD IMPLEMENTATIONS
*/

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

uint32 *extractMartrices(TIFF *tif, string fileName) {
  TIFF *firstTimePoint = TIFFOpen(fileName.c_str(), "w");
  if (firstTimePoint) {
    tdata_t buf;

    uint32 height, width, photo;
    short samplesPerPixel, bitsPerSample;
    tsize_t scanLineSize;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

    uint32 *currentTimePoint = new uint32[width * height];

    TIFFSetField(firstTimePoint, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(firstTimePoint, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(firstTimePoint, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
    TIFFSetField(firstTimePoint, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
    TIFFSetField(firstTimePoint, TIFFTAG_PHOTOMETRIC, photo);
    cout << "\nTIMEPOINT 1 .tif info:" << endl;
    printf(
        "width = %d\nheight = %d\nsamplesPerPixel = %d\nbitsPerSample = %d\n\n",
        width, height, samplesPerPixel, bitsPerSample);
    scanLineSize = TIFFScanlineSize(tif);
    buf = _TIFFmalloc(scanLineSize);
    cout << "TIFF SCANLINE SIZE IS " << scanLineSize << " bits" << endl;
    // printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height,
    // width,TIFFScanlineSize(tif));
    for (uint32 row = 0; row < height; row++) {
      if (TIFFReadScanline(tif, buf, row, 0) != -1) {
        memcpy(&currentTimePoint[row * width], buf, scanLineSize);
        if (TIFFWriteScanline(firstTimePoint, buf, row, 0) == -1) {
          cout << "ERROR WRITING SCANLINE" << endl;
          exit(-1);
        }
      } else {
        cout << "ERROR READING SCANLINE" << endl;
        exit(-1);
      }
    }
    TIFFClose(firstTimePoint);
    _TIFFfree(buf);
    return currentTimePoint;
  } else {
    cout << "COULD NOT CREATE FIRST TIMEPOINT TIFF" << endl;
    exit(-1);
  }
}

uint32 *extractMartrices(TIFF *tif) {

  uint32 height, width;
  tdata_t buf;

  vector<uint32 *> currentPlane;
  tsize_t scanLineSize;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

  uint32 *currentTimePoint = new uint32[width * height];
  scanLineSize = TIFFScanlineSize(tif);
  buf = _TIFFmalloc(scanLineSize);

  // printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height,
  // width,TIFFScanlineSize(tif));
  for (uint32 row = 0; row < height; row++) {
    if (TIFFReadScanline(tif, buf, row, 0) != -1) {
      memcpy(&currentTimePoint[row * width], buf, scanLineSize);
    } else {
      cout << "ERROR READING SCANLINE" << endl;
      exit(-1);
    }
  }
  _TIFFfree(buf);
  return currentTimePoint;
}

__global__ void normalize(uint32 *flatMatrix, float *normals, uint32 min,
                          uint32 max, long size) {
  int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  long globalID = blockID * blockDim.x + threadIdx.x;
  int stride = gridDim.x * gridDim.y * blockDim.x;
  float currentValue = 0;
  float dmin = static_cast<float>(min);
  float dmax = static_cast<float>(max);
  while (globalID < size) {
    if (flatMatrix[globalID] != 0) {
      currentValue = static_cast<float>(flatMatrix[globalID]) - dmin;
      currentValue /= (dmax - dmin);
    }
    normals[globalID] = 1.0f / (1.0f + expf((-10.0f * currentValue) + 7.5));
    globalID += stride;
  }
}
