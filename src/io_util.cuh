#ifndef IO_UTIL_CUH
#define IO_UTIL_CUH

#include "common_includes.h"
#include "cuda_zebra.cuh"

void parseArgs(const int &numArgs, char** args, unsigned int &k, unsigned int &iterations,
  double &learningRate, double &threshold, float &sigmoidTuner, std::string &baseDirectory);

std::string createFourCharInt(int i);
void extractMartrices(TIFF *tif, uint32* &imageMatrix, unsigned int width, unsigned int height, unsigned int scanLineSize);
uint32* readTiffVideo(std::string videoDirectoryPath, unsigned int &width, unsigned int &height, unsigned int &numTimePoints, std::string &baseName);
int createKey(float* &mtx, bool* &key, unsigned int numTimePoints, unsigned long numPixels);

void createSpatialImages(std::string outDir, std::string firstTimePointLocation, std::string baseName, int k,
  unsigned int width, unsigned int height, float* W, bool* key, uint32 &samplesPerPixel, uint32 &bitsPerSample, uint32 &photo);
void createKVideos(std::string outDir, std::string baseName, int k, unsigned int width, unsigned int height,
  unsigned int numTimePoints, float* W, float* H, bool* key, uint32 samplesPerPixel, uint32 bitsPerSample, uint32 photo);
void createVisualization(std::string videoDirectoryPath, int k, unsigned int width, unsigned int height,
  unsigned int numTimePoints, float* W, float* H, bool* key, std::string baseName);
#endif /* IO_UTIL_CUH */
