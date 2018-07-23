#ifndef IO_UTIL_H
#define IO_UTIL_H

#include "common_includes.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

std::string createFourCharInt(int i);
void extractMartrices(TIFF *tif, uint32* &imageMatrix, unsigned int width, unsigned int height, unsigned int scanLineSize);
uint32* readTiffVideo(std::string videoDirectoryPath, unsigned int &width, unsigned int &height, unsigned int &numTimePoints, std::string &baseName);
int createKey(float* &matrix, bool* &key, unsigned int numTimePoints, unsigned long numPixels);

void createVisualization(std::string videoDirectoryPath, int k, unsigned int width, unsigned int height,
  unsigned int numTimePoints, float* W, float* H, float* matrix, bool* key, std::string baseName);
#endif /* IO_UTIL_H */
