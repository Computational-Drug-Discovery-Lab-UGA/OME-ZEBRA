#include "common_includes.h"
#include "io_util.h"
#include "cuda_zebra.cuh"
#include "matrix.h"


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
  performNNMF(W, H, minimizedVideo, k, height*width, numTimePoints);
  createVisualization(baseDirectory,k, width, height, numTimePoints, W, H, minimizedVideo, key, baseName);

  return 0;
}
