#ifndef IO_UTIL_H
#define IO_UTIL_H

#include "common_includes.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

std::string createFourCharInt(int i);
void extractMartrices(TIFF *tif, uint32* &imageMatrix, unsigned int width, unsigned int height, unsigned int scanLineSize);
uint32* readTiffVideo(std::string videoDirectoryPath, unsigned int &width, unsigned int &height, unsigned int &numTimePoints);
void createSpatialTiffs(std::string directoryWithNNMF, int k, unsigned int width, unsigned int height, unsigned int numTimePoints);
void createKVideo(std::string filePath, unsigned int width, unsigned int height, unsigned int numTimePoints);
void createVisualization(std::string videoDirectoryPath, int k, unsigned int width, unsigned int height, unsigned int numTimePoints);
#endif /* IO_UTIL_H */
