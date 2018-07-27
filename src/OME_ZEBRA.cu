#include "common_includes.h"
#include "io_util.cuh"
#include "cuda_zebra.cuh"

int main(int argc, char *argv[]) {
  if(argc != 2){
    std::cout << "Usage: ./exe <directory of timepoint tifs>";
    exit(-1);
  }
  std::string baseDirectory = argv[1];
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int numTimePoints = 0;
  std::string baseName = "";
  uint32* tifVideo = readTiffVideo(baseDirectory, width, height, numTimePoints, baseName);
  float* normVideo = executeNormalization(tifVideo, width*height*numTimePoints);

  std::string testImage = baseDirectory + "test.tif";
  float max = std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  uint32* pixRow = new uint32[width];
  for(int i = 0; i < height*width; ++i){
    if(normVideo[i*numTimePoints + 2] < min){
      min = normVideo[i*numTimePoints + 2];
    }
    if(normVideo[i*numTimePoints + 2] > max){
      max = normVideo[i*numTimePoints + 2];
    }
  }
  TIFF *tif = TIFFOpen(testImage.c_str(), "w");
  if(tif){
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);

    for(int row = 0; row < height; ++row){
      for(int col = 0; col < width; ++col){
        pixRow[col] =  (uint32) (UINT32_MAX*(normVideo[(row*width + col)*numTimePoints + 2] - min)/(max-min));
      }
      if (TIFFWriteScanline(tif, pixRow, row, 0) != 1) {
        std::cout << "ERROR WRITING FIRST TIMEPOINT" << std::endl;
        exit(-1);
      }
    }
  }



  delete[] tifVideo;
  unsigned long numPixelsWithValues = 0;
  bool* key = generateKey(width*height, numTimePoints, normVideo, numPixelsWithValues);
  float* minimizedVideo;
  if(numPixelsWithValues != height*width){
    minimizedVideo = minimizeVideo(height*width, numPixelsWithValues, numTimePoints, normVideo, key);
    delete[] normVideo;
  }
  else{
    minimizedVideo = normVideo;
  }

  unsigned int k = 10;
  float* W = new float[k*width*height];
  float* H = new float[k*numTimePoints];
  //W & H are not currently and neither are any c files
  performNNMF(W, H, minimizedVideo, k, height*width, numTimePoints, baseDirectory);
  createVisualization(baseDirectory,k, width, height, numTimePoints, W, H, key, baseName);

  return 0;
}
