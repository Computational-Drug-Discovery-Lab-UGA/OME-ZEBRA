#include "common_includes.h"
#include "io_util.cuh"
#include "cuda_zebra.cuh"

int main(int argc, char *argv[]) {
  cuInit(0);

  unsigned int k = 2;
  unsigned int iterations = 1000;
  double learningRate = .1;
  double threshHold = 1e-8;
  float sigmoidTuner = 9.0;
  std::string baseDirectory = "";

  parseArgs(argc, argv, k, iterations, learningRate, threshHold, sigmoidTuner, baseDirectory);

  if(baseDirectory.substr(baseDirectory.length() - 1,1) != "/") baseDirectory += "/";
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int numTimePoints = 0;
  std::string baseName = "";
  uint32* tifVideo = readTiffVideo(baseDirectory, width, height, numTimePoints, baseName);
  float* normVideo = executeNormalization(tifVideo, width*height*numTimePoints, sigmoidTuner);
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
  performNNMF(W, H, minimizedVideo, k, iterations, learningRate, threshHold, height*width, numTimePoints, baseDirectory);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  cuInit(0);
  createVisualization(baseDirectory, k, width, height, numTimePoints, W, H, key, baseName);
  cudaDeviceReset();
  return 0;
}
