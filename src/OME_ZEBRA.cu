#include "common_includes.h"
#include "io_util.cuh"
#include "cuda_zebra.cuh"

int main(int argc, char *argv[]) {
  cuInit(0);

  unsigned int k;
  unsigned int iterations;
  double learningRate;
  double threshHold;
  std::string baseDirectory;

  if(argc < 2){
    std::cout << "Usage: ./exe <directory of timepoint tifs>";
    exit(-1);
  }

  else if (argc == 11){

    for(int i = 1; i < argc; ++i) {

        if(std::string(argv[i]) == "-d") {

          i++;
          baseDirectory = argv[i];

        }

        else if(std::string(argv[i]) == "-k") {

          i++;
          k = std::stoi(argv[i]);

        }

        else if(std::string(argv[i]) == "-i") {

          i++;
          iterations = std::stoi(argv[i]);

        }
        else if(std::string(argv[i]) == "-l") {

          i++;
          learningRate = std::stod(argv[i]);

        }
        else if(std::string(argv[i]) == "-t") {

          i++;
          threshHold = std::stod(argv[i]);

        }

    }

  }
  else{

    std::cout << "Please use all arguments listed in documentation" << '\n';

  }

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
  performNNMF(W, H, minimizedVideo, k, iterations, learningRate, threshHold, height*width, numTimePoints, baseDirectory);
  cudaDeviceSynchronize();
  cudaDeviceReset();
  cuInit(0);
  createVisualization(baseDirectory, k, width, height, numTimePoints, W, H, key, baseName);
  cudaDeviceReset();
  return 0;
}
