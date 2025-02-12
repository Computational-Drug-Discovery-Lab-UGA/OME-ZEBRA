#include "io_util.cuh"

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
IO HELPERS
*/

void parseArgs(const int &numArgs, char** args, unsigned int &k, unsigned int &iterations,
  double &learningRate, double &threshold, float &sigmoidTuner, std::string &baseDirectory){

  std::map<std::string, int> parameters;
  parameters["-k"] = 0;
  parameters["-i"] = 1;
  parameters["-l"] = 2;
  parameters["-t"] = 3;
  parameters["-s"] = 4;
  parameters["-h"] = 5;
  parameters["-help"] = 6;
  if(numArgs < 2){
    std::cout << "Usage: ./exe <directory of timepoint tifs>\nfor help ./exe -h"<<std::endl;
    exit(-1);
  }
  else if (numArgs >= 2){
    if(parameters[args[1]] < 5){
      baseDirectory = args[1];
    }
    else{
      std::ifstream readme;
      std::string line = "";
      readme.open("README.txt");
      if(readme.is_open()){
        while(getline(readme, line)){
          std::cout << line << std::endl;
        }
        readme.close();
      }
      exit(0);
    }
    for(int i = 2; i < numArgs; ++i) {
      switch(parameters[args[i]]){
        case 0:
          k = std::stoi(args[++i]);
          break;
        case 1:
          iterations = std::stoi(args[++i]);
          break;
        case 2:
          learningRate = std::stod(args[++i]);
          break;
        case 3:
          threshold = std::stod(args[++i]);
          break;
        case 4:
          sigmoidTuner = std::stof(args[++i]);
          break;
        default:
          std::cout<<"Please use all arguments listed in documentation"<<std::endl;
          exit(-1);
      }
    }
  }
}

std::string createFourCharInt(int i) {
  std::string strInt;
  if (i < 10) {
    strInt = "000" + std::to_string(i);
  } else if (i < 100) {
    strInt = "00" + std::to_string(i);
  } else if (i < 1000) {
    strInt = "0" + std::to_string(i);
  } else {
    strInt = std::to_string(i);
  }
  return strInt;
}
void extractMartrices(TIFF *tif, uint32* &imageMatrix, unsigned int width, unsigned int height, unsigned int scanLineSize) {
  tdata_t buf;
  buf = _TIFFmalloc(scanLineSize);

  for (uint32 row = 0; row < height; row++) {
    if (TIFFReadScanline(tif, buf, row, 0) != -1) {
      memcpy(&imageMatrix[row * width], buf, scanLineSize);
    } else {
      std::cout << "ERROR READING SCANLINE" << std::endl;
      exit(-1);
    }
  }
  _TIFFfree(buf);
}
uint32* readTiffVideo(std::string videoDirectoryPath, unsigned int &width, unsigned int &height, unsigned int &numTimePoints, std::string &baseName){
  DIR* dir;
  if (NULL == (dir = opendir(videoDirectoryPath.c_str()))){
    printf("Error : Failed to open input directory %s\n",videoDirectoryPath.c_str());
    exit(-1);
  }
  struct dirent* in_file;
  std::vector<uint32*> videoVector;
  unsigned int scanLineSize;
  std::cout<<"reading tif timepoint files from "<<videoDirectoryPath<<std::endl;
  std::vector<std::string> fileNames;
  while((in_file = readdir(dir)) != NULL){
    std::string currentFileName = in_file->d_name;
    if (currentFileName == "." || currentFileName == ".." || currentFileName.length() < 5||
      currentFileName.substr(currentFileName.length() - 3) != "tif") continue;
    //TODO check if it is a tif
    if (numTimePoints == 0) {
      baseName = currentFileName.substr(currentFileName.find_last_of("/") + 1, currentFileName.length() - 8);
    }
    currentFileName = videoDirectoryPath + currentFileName;
    fileNames.push_back(currentFileName);
    ++numTimePoints;
  }
  closedir(dir);
  std::sort(fileNames.begin(), fileNames.end());
  for(int i = 0; i < numTimePoints; ++i){
    TIFF *tif = TIFFOpen(fileNames[i].c_str(), "r");
    if (tif) {
      if (i == 0) {
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        scanLineSize = TIFFScanlineSize(tif);
      }
      uint32 *tpMatrix = new uint32[height*width];
      extractMartrices(tif, tpMatrix, width, height, scanLineSize);
      videoVector.push_back(tpMatrix);
      TIFFClose(tif);
    }
    else{
      std::cout<<"READING IN TIFF DIRECTORY FAILED AT TP = "<<numTimePoints<<std::endl;
      exit(-1);
    }
  }
  uint32* videoMatrix = new uint32[height*width*numTimePoints];
  for(int r = 0; r < height*width; ++r){
    for(int c = 0; c < numTimePoints; ++c){
      videoMatrix[r*numTimePoints + c] = videoVector[c][r];
    }
  }
  printf("width = %d::height = %d::numTimePoints = %d\nbaseName = %s\n", width, height, numTimePoints, baseName.c_str());
  return videoMatrix;
}

int createKey(float* &mtx, bool* &key, unsigned int numTimePoints, unsigned long numPixels){
  int nonZeroCounter = 0;
  int lastGoodIndex = 0;
  for (unsigned i = 0; i < numPixels; i++) {
    nonZeroCounter = 0;
    for (unsigned j = 0; j < numTimePoints; j++) {
      if (mtx[(numTimePoints * i) + j] != 0.0f) {
        nonZeroCounter++;
        break;
      }
    }
    if (nonZeroCounter != 0) {
      key[i] = true;
      lastGoodIndex++;
    }
  }
  return lastGoodIndex;
}

void createSpatialImages(std::string outDir, std::string firstTimePointLocation, std::string baseName, int k,
  unsigned int width, unsigned int height, float* W, bool* key, uint32 &samplesPerPixel, uint32 &bitsPerSample, uint32 &photo){

  uint32 max = 0;
  uint32 min = UINT32_MAX;
  uint32 ***kMatrix = new uint32**[k];
  for (int i = 0; i < k; ++i) {
    kMatrix[i] = new uint32*[height];
    for (int ii = 0; ii < height; ++ii) {
      kMatrix[i][ii] = new uint32[width];
    }
  }
  TIFF *tif = TIFFOpen(firstTimePointLocation.c_str(), "r");
  if(tif){
    tdata_t buf;
    tsize_t scanLineSize;
    uint32 row;
    std::vector<uint32 *> currentPlane;

    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

    scanLineSize = TIFFScanlineSize(tif);
    buf = _TIFFmalloc(scanLineSize);

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
        std::cout << "ERROR READING SCANLINE" << std::endl;
        exit(-1);
      }
    }
    printf("first timepoint - (uint32) min = %d, max = %d\n",min,max);
    TIFFClose(tif);
    _TIFFfree(buf);
  }
  else{
    std::cout<<"cannot open "<<firstTimePointLocation<<std::endl;
    exit(-1);
  }

  int largest = 0;
  float largestValue = 0.0f;
  float currentValue = 0.0f;
  for(int row = 0; row < height; ++row){
    for (int col = 0; col < width; ++col) {
      for (int kFocus = 0; kFocus < k; kFocus++) {
        kMatrix[kFocus][row][col] -= min;
      }
      if(key[row*width + col]){
        largest = 0;
        largestValue = 0.0f;
        currentValue = 0.0f;
        for (int kFocus = 0; kFocus < k; kFocus++) {
          currentValue = W[(row*width + col)*k + kFocus];
          if (largestValue < currentValue) {
            largest = kFocus;
            largestValue = currentValue;
          }
        }
        kMatrix[largest][row][col] += (max - min) / 2;
      }
    }
  }
  for (int kFocus = 0; kFocus < k; ++kFocus) {
    std::string fileName = outDir + baseName + "_spacial_K" + std::to_string(k) + "_k" + std::to_string(kFocus) + ".tif";
    TIFF *resultTif = TIFFOpen(fileName.c_str(), "w");
    if (resultTif) {
      TIFFSetField(resultTif, TIFFTAG_IMAGEWIDTH, width);
      TIFFSetField(resultTif, TIFFTAG_IMAGELENGTH, height);
      TIFFSetField(resultTif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
      TIFFSetField(resultTif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
      TIFFSetField(resultTif, TIFFTAG_PHOTOMETRIC, photo);
      for(uint32 row = 0; row < height; ++row){
        if (TIFFWriteScanline(resultTif, kMatrix[kFocus][row], row, 0) != 1) {
          std::cout << "ERROR WRITING FIRST TIMEPOINT" << std::endl;
          exit(-1);
        }
      }
      std::cout<<fileName<<" has been created"<<std::endl;
      TIFFClose(resultTif);
    }
    else{
      std::cout<<"COULD NOT OPEN TIF"<<std::endl;
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
}
void createKVideos(std::string outDir, std::string baseName, int k, unsigned int width, unsigned int height,
  unsigned int numTimePoints, float* W, float* H, bool* key, uint32 samplesPerPixel, uint32 bitsPerSample, uint32 photo){

  float* wColDevice;
  float* hRowDevice;
  float* resultTransposeDevice;
  float* resultTranspose = new float[height*width*numTimePoints];
  float* wCol = new float[height*width];
  CudaSafeCall(cudaMalloc((void**)&wColDevice, height*width*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&hRowDevice, numTimePoints*sizeof(float)));
  CudaSafeCall(cudaMalloc((void**)&resultTransposeDevice, width*height*numTimePoints*sizeof(float)));

  dim3 grid = {1,1,1};
  dim3 block = {1,1,1};
  float greatestTPK = 0.0f;
  int signatureTimePoint;
  getFlatGridBlock(height*width*numTimePoints, grid, block);
  for(int kFocus = 0; kFocus < k; ++kFocus){

    for(int w = 0; w < height*width; ++w){
      wCol[w] = W[w*k + kFocus];
    }
    CudaSafeCall(cudaMemcpy(wColDevice, wCol, height*width*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(hRowDevice, H + (kFocus*numTimePoints), numTimePoints*sizeof(float), cudaMemcpyHostToDevice));
    multiplyRowColumn<<<grid,block>>>(wColDevice, hRowDevice, resultTransposeDevice, height*width, numTimePoints);
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(resultTranspose, resultTransposeDevice, width*height*numTimePoints*sizeof(float), cudaMemcpyDeviceToHost));

    greatestTPK = 0.0f;
    for(int tp = 0; tp < numTimePoints; ++tp){
      if(H[kFocus*numTimePoints + tp] > greatestTPK){
        greatestTPK = H[kFocus*numTimePoints + tp];
        signatureTimePoint = tp;
      }
    }
    std::string newTif = outDir + baseName + "_temporal_K" + std::to_string(k) +  "_k" + std::to_string(kFocus) +
      "_tp" + createFourCharInt(signatureTimePoint) + ".tif";

    TIFF *tpfTif = TIFFOpen(newTif.c_str(), "w");
    if(tpfTif){
      TIFFSetField(tpfTif, TIFFTAG_IMAGEWIDTH, width);
      TIFFSetField(tpfTif, TIFFTAG_IMAGELENGTH, height);
      TIFFSetField(tpfTif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
      TIFFSetField(tpfTif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
      TIFFSetField(tpfTif, TIFFTAG_PHOTOMETRIC, photo);

      for(uint32 row = 0; row < height; ++row){
        if(TIFFWriteScanline(tpfTif, resultTranspose + ((signatureTimePoint*width*height) + (row*width)), row, 0) != 1) {
          std::cout << "ERROR WRITING FIRST TIMEPOINT" << std::endl;
          exit(-1);
        }
      }
      TIFFClose(tpfTif);
    }
    else{
      std::cout<<"COULD NOT CREATE "<<newTif<<std::endl;
      exit(-1);
    }
    std::cout<<newTif<<" has been created"<<std::endl;
  }

  CudaSafeCall(cudaFree(wColDevice));
  CudaSafeCall(cudaFree(hRowDevice));
  CudaSafeCall(cudaFree(resultTransposeDevice));
  delete[] wCol;
  delete[] resultTranspose;
}
void createVisualization(std::string videoDirectoryPath, int k, unsigned int width, unsigned int height,
  unsigned int numTimePoints, float* W, float* H, bool* key, std::string baseName){

  std::string outDir = videoDirectoryPath + "out/";
  if(mkdir(outDir.c_str(), 0777) == -1){
    std::cout<<"CANNOT CREATE "<<outDir<<std::endl;
  }
  DIR* dir;
  std::string firstTimePointLocation = "";
  if (NULL == (dir = opendir(videoDirectoryPath.c_str()))){
    printf("Error : Failed to open input directory %s\n",videoDirectoryPath.c_str());
    exit(-1);
  }
  struct dirent* in_file;
  std::string currentFileName = "";
  while((in_file = readdir(dir)) != NULL){
    if (in_file->d_name == "." || in_file->d_name == "..") continue;
    currentFileName = in_file->d_name;
    if(currentFileName.find("0000.tif") != std::string::npos){
      firstTimePointLocation = videoDirectoryPath + currentFileName;
      break;
    }
  }
  closedir(dir);

  uint32 samplesPerPixel = 0, bitsPerSample = 0, photo = 0;

  createSpatialImages(outDir, firstTimePointLocation, baseName, k, width, height, W, key, samplesPerPixel, bitsPerSample, photo);
  createKVideos(outDir, baseName, k, width, height, numTimePoints, W, H, key, samplesPerPixel, bitsPerSample, photo);
}
