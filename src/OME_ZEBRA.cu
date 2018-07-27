#include "common_includes.h"
#include "io_util.cuh"
#include "cuda_zebra.cuh"

int main(int argc, char *argv[]) {
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
  float* W;
  float* H;
  //W & H are not currently and neither are any c files
  performNNMF(W, H, minimizedVideo, k, height*width, numTimePoints, baseDirectory);
  createVisualization(baseDirectory,k, width, height, numTimePoints, W, H, key, baseName);
  cudaDeviceReset();
  return 0;
}

//THIS COMMENTED OUT CODE IS USED TO PRING AN IMAGE FROM A FLOAT VIDEO matrix
//specify timepoint where 511 is
// DIR* dir;
// std::string firstTimePointLocation = "";
// if (NULL == (dir = opendir(baseDirectory.c_str()))){
//   printf("Error : Failed to open input directory %s\n",baseDirectory.c_str());
//   exit(-1);
// }
// struct dirent* in_file;
// std::string currentFileName = "";
// while((in_file = readdir(dir)) != NULL){
//   if (in_file->d_name == "." || in_file->d_name == "..") continue;
//   currentFileName = in_file->d_name;
//   if(currentFileName.find("0000.tif") != std::string::npos){
//     firstTimePointLocation = baseDirectory + currentFileName;
//     break;
//   }
// }
// closedir(dir);
// TIFF *tif = TIFFOpen(firstTimePointLocation.c_str(), "r");
// uint32 samplesPerPixel, bitsPerSample, photo;
// if(tif){
//   TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
//   TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
//   TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
//   TIFFClose(tif);
// }
// else{
//   std::cout<<"cannot open "<<firstTimePointLocation<<std::endl;
//   exit(-1);
// }
//
// std::string testImage = baseDirectory + "test.tif";
// float max = std::numeric_limits<float>::min();
// float min = std::numeric_limits<float>::max();
// uint32* pixRow = new uint32[width];
// for(int i = 0; i < height*width; ++i){
//   if(minimizedVideo[i*numTimePoints + 511] < min){
//     min = minimizedVideo[i*numTimePoints + 511];
//   }
//   if(minimizedVideo[i*numTimePoints + 511] > max){
//     max = minimizedVideo[i*numTimePoints + 511];
//   }
// }
// printf("tp = %d, (float) - min = %f, max = %f\n",511, min, max);
// TIFF *testtif = TIFFOpen(testImage.c_str(), "w");
// if(testtif){
//   TIFFSetField(testtif, TIFFTAG_IMAGEWIDTH, width);
//   TIFFSetField(testtif, TIFFTAG_IMAGELENGTH, height);
//   TIFFSetField(testtif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
//   TIFFSetField(testtif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
//   TIFFSetField(testtif, TIFFTAG_PHOTOMETRIC, photo);
//   uint32 currentValue = 0;
//   for(int row = 0; row < height; ++row){
//     for(int col = 0; col < width; ++col){
//       currentValue =  (uint32) (UINT32_MAX*(minimizedVideo[(row*width + col)*numTimePoints + 100] - min)/(max-min));
//       pixRow[col] = currentValue;
//     }
//     if (TIFFWriteScanline(testtif, pixRow, row, 0) != 1) {
//       std::cout << "ERROR WRITING FIRST TIMEPOINT" << std::endl;
//       exit(-1);
//     }
//   }
//   TIFFClose(testtif);
// }
// exit(0);
