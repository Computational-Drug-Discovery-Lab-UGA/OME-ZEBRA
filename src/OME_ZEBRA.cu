#include "common_includes.h"
#include "io_util.cuh"
#include "cuda_zebra.cuh"

int main(int argc, char *argv[]) {
  cuInit(0);
  if(argc < 2 || argc > 3){
    std::cout << "Usage: ./exe <directory of timepoint tifs>";
    exit(-1);
  }
  unsigned int k;
  if(argc == 3) k = std::stoi(argv[2]);
  else{
    k = 2;
  }
  std::string baseDirectory = argv[1];
  if(baseDirectory.substr(baseDirectory.length() - 1,1) != "/") baseDirectory += "/";
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
  float* W = new float[height*width*k];
  float* H = new float[k*numTimePoints];
  //NOTE minimized video is deleted in performNNMF
  performNNMF(W, H, minimizedVideo, k, height*width, numTimePoints, baseDirectory);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  cuInit(0);
  createVisualization(baseDirectory,k, width, height, numTimePoints, W, H, key, baseName);
  cudaDeviceReset();
  return 0;
}
