#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include "tiffio.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace std;


int main(int argc, char *argv[]) {

    if(argc!=2) {
      cout << "Usage: ./exe <file>";
      return 1;
    }
    else {
      string tifName = argv[1];
      string tifFile = "data/out/" + tifName + "/" + tifName + "_TP1.tif";
      string wFileLocation = "data/out/" + tifName + "/" + tifName + "_W.txt";

      TIFF* tif = TIFFOpen(tifFile.c_str(), "r");
      string fileName = "data/out/" + tifName + "/" + tifName  + "_RESULT.tif";
      if (tif) {
        TIFF* resultTif = TIFFOpen(fileName.c_str(), "w");
        if(resultTif){
          tdata_t buf;
          uint32 row;
          uint32 config;
          vector<uint32*> currentPlane;

          uint32 height, width, samplesPerPixel, bitsPerSample, photo, scanLineSize;

          TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
          TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
          TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
          TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
          TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
          TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &scanLineSize);

          cout<<"\nTIMEPOINT 1 .tif info:"<<endl;
          printf("width = %d\nheight = %d\nsamplesPerPixel = %d\nbitsPerSample = %lo\nscanLineSize = %d\n\n",width,height,samplesPerPixel,bitsPerSample,photo,scanLineSize);

          TIFFSetField(resultTif, TIFFTAG_IMAGEWIDTH, width);
          TIFFSetField(resultTif, TIFFTAG_IMAGELENGTH, height);
          TIFFSetField(resultTif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
          TIFFSetField(resultTif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
          TIFFSetField(resultTif, TIFFTAG_PHOTOMETRIC, photo);
          TIFFSetField(resultTif, TIFFTAG_ROWSPERSTRIP, scanLineSize);
          buf = _TIFFmalloc(TIFFScanlineSize(tif));
          uint32* data;
          uint32 max = 0;

          //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
          for (row = 0; row < height; ++row){
            TIFFReadScanline(tif, buf, row);
            data=(uint32*)buf;
            for(int col = 0; col < width; ++col){
              if(data[col] > max) max = data[col];
            }
          }
          int currentPixel = 0;
          ifstream wFile(wFileLocation);
          float seizure, isNot;
          string currentLine;
          if(wFile.is_open()){
            for (row = 0; row < height; ++row){
              TIFFReadScanline(tif, buf, row);
              data=(uint32*)buf;
              for(int col = 0; col < width; ++col){
                getline(wFile, currentLine);
                stringstream ss;
                ss<<currentLine;
                ss>>seizure;
                ss>>isNot;
                //if(seizure > isNot) data[col] = max;
                data[col]+=1;
              }
              if(TIFFWriteScanline(resultTif, data, row) != 1){
                cout<<"ERROR WRITING FIRST TIMEPOINT"<<endl;
                exit(-1);
              }
              currentPixel++;
            }
          }
          TIFFClose(resultTif);
          _TIFFfree(buf);
        }
        TIFFClose(tif);
      }
      else{
        cout<<"COULD NOT OPEN"<<argv[1]<<endl;
        return 1;
      }
    }
    return 0;
}
