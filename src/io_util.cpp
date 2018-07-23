#include "io_util.h"

std::string createFourCharInt(int i) {
  std::string strInt;
  if (i < 10) {
    strInt = "000" + to_string(i);
  } else if (i < 100) {
    strInt = "00" + to_string(i);
  } else if (i < 1000) {
    strInt = "0" + to_string(i);
  } else {
    strInt = to_string(i);
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

uint32* readTiffVideo(std::string videoDirectoryPath, unsigned int &width, unsigned int &height, unsigned int &numTimePoints){
  DIR* dir;
  if (NULL == (dir = opendir(videoDirectoryPath.c_str()))){
      printf("Error : Failed to open input directory %s\n",videoDirectoryPath);
      exit(-1);
    }
  }
  struct dirent* in_file;
  std::string currentFileName = "";
  vector<uint32*> videoVector;
  while(in_file = readDir(dir)){
    if (in_file->d_name != "." || in_file->d_name != "..") continue;
    currentFileName = in_file->d_name;
    //TODO check if it is a tif
    TIFF *tif = TIFFOpen(currentFileName.c_str(), "r");
    if (tif) {
      if (numTimePoints == 0) {
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        scanLineSize = TIFFScanlineSize(tif);
      }

      uint32 *tpMatrix = new uint32[height*width];
      extractMartrices(tif, tpMatrix, width, height, scanLineSize);
      videoVector.push_back(flatMatrix);
      TIFFClose(tif);
    }
    else{
      std::cout<<"READING IN TIFF DIRECTORY FAILED AT TP = "<<numTimePoints<<std::endl;
    }
    ++numTimePoints;
  }
  closedir(dir);
  uint32* videoMatrix = new uint32[height*width*numTimePoints];
  for(int i = 0; i < numTimePoints; ++i){
    memcpy(&videoMatrix[i*height*width], videoVector[i], height*width*sizeof(uint32));
  }
  for(int r = 0; r < height*width; ++r){
    for(int c = 0; c < numTimePoints; ++c){
      videoMatrix[r*numTimePoints + c] = videoVector[c][r];
    }
  }
  return videoMatrix;
}

void createSpatialTiffs(float* w, int k, unsigned int width, unsigned int height, unsigned int numTimePoints){


}

void createKVideo(float* w, float* h, bool* key, unsigned int width, unsigned int height, unsigned int numTimePoints){


}

void createVisualization(std::string videoDirectoryPath, int k, unsigned int width, unsigned int height, unsigned int numTimePoints){
  std::wFileLocation = videoDirectoryPath + "out/NNMF.nmf_W.txt";
  std::wFileLocation = videoDirectoryPath + "out/NNMF.nmf_H.txt";
  std::keyFileLocation = videoDirectoryPath + "out/key.csv";
  DIR* dir;
  std::firstTimePointLocation = "";
  if (NULL == (dir = opendir(videoDirectoryPath.c_str()))){
      printf("Error : Failed to open input directory %s\n",videoDirectoryPath);
      exit(-1);
    }
  }
  struct dirent* in_file;
  std::string currentFileName = "";
  while(in_file = readDir(dir)){
    if (in_file->d_name != "." || in_file->d_name != "..") continue;
    currentFileName = in_file->d_name;
    if(currentFileName.find("0000.tif") != std::string::npos){
      firstTimePointLocation = currentFileName;
      break;
    }
  }

}
