/*
Created and edited by Jackson Parker and Godfrey Hendrix
This script is used to create a visualized tif image of the output
of NNMF run on a tif directory. The necessary inputs include a tif image
of the first timepoint, the w matrix of NNMF output and a  key csv that represents
the if a pixel row is all 0.
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include "tiffio.h"
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>

using namespace std;


int main(int argc, char *argv[]) {
    if(argc < 4) {
      cout << "Usage: ./exe <file> <timeGroup> <k> <kFocus(optional will = 0 if not set)>";
      return 1;
    }
    else {
      int k = 2;
      int tG = 0;
      int kFocus = 0;
      string wFileLocation;
      string tifName = argv[1];
      string tifFile = "data/out/" + tifName + "/" + tifName + ".ome0000.tif";
      TIFF* tif = TIFFOpen(tifFile.c_str(), "r");
      istringstream argT(argv[2]);
      argT >> tG;
      wFileLocation = "data/out/" + tifName + "/NNMF_" + to_string(tG)  + ".nmf_W.txt";
      string keyFileLocation = "data/out/" + tifName + "/" + "key_" + to_string(tG) + ".csv";

      istringstream argK(argv[3]);
      argK >> k;
      if(argc == 5){
        istringstream argKFocus(argv[4]);
        argKFocus >> kFocus;
      }
      string fileName = "data/out/" + tifName + "/" + tifName + "_" + to_string(tG) + "_" + to_string(k) + "_" + to_string(kFocus) + "_RESULT.tif";
      cout<<wFileLocation<<endl;
      cout<<tifFile<<endl;
      cout<<keyFileLocation<<endl;
      cout<<fileName<<endl;
      cout<<"Time group = "<<tG<<endl;
      cout<<"k = "<<k<<endl;
      cout<<"kFocus = "<<kFocus<<endl;
      if(k <= kFocus){
        cout<<"ERROR this condition must be met kFocus < k"<<endl;
        exit(-1);
      }
      if (tif) {
        TIFF* resultTif = TIFFOpen(fileName.c_str(), "w");

        if(resultTif){
          tdata_t buf;
          tsize_t scanLineSize;
          uint32 row;
          uint32 config;
          vector<uint32*> currentPlane;

          uint32 height, width, samplesPerPixel, bitsPerSample, photo;

          TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
          TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
          TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
          TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
          TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

          TIFFSetField(resultTif, TIFFTAG_IMAGEWIDTH, width);
          TIFFSetField(resultTif, TIFFTAG_IMAGELENGTH, height);
          TIFFSetField(resultTif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
          TIFFSetField(resultTif, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
          TIFFSetField(resultTif, TIFFTAG_PHOTOMETRIC, photo);

          scanLineSize = TIFFScanlineSize(tif);
          buf = _TIFFmalloc(scanLineSize);

          uint32 max = 0;
          uint32 min = 4294967295;
          uint32* data = new uint32[width];
          cout<<"starting visualization prep"<<endl;
          ifstream wFile(wFileLocation);
          ifstream keyFile(keyFileLocation);
          float* kArray = new float[k];
          string currentLine;
          //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
          for (row = 0; row < height; ++row){
            if(TIFFReadScanline(tif, buf, row, 0) != -1){
              memcpy(data, buf, scanLineSize);
              for(int col = 0; col < width; ++col){
                if(data[col] > max) max = data[col];
                if(data[col] < min && data[col] != 0) min = data[col];
              }
            }
            else{
              cout<<"ERROR READING SCANLINE"<<endl;
              exit(-1);
            }
          }
          currentLine = "";
          string temp  = "";
          bool currentKey = true;
          if(wFile.is_open() && keyFile.is_open()){
            cout<<"key and W nmf file are open"<<endl;
            for (row = 0; row < height; ++row){
              if(TIFFReadScanline(tif, buf, row, 0) != -1){
                memcpy(data, buf, scanLineSize);
                for(int col = 0; col < width; ++col){
                  getline(keyFile, temp);
                  istringstream keyss(temp);
                  keyss >> currentKey;
                  data[col] -= min;
                  if(currentKey){
                    getline(wFile, currentLine);
                    istringstream ss(currentLine);
                    for (int kIterator = 0; kIterator < k; kIterator++) {

                      ss>>kArray[kIterator];
                      //cout<<kArray[kIterator]<<" ";
                    }
                    //cout<<endl;
                    bool isLargest = true;
                    for (int kIterator = 0; kIterator < k ; kIterator++) {

                      if (kArray[kFocus] < kArray[kIterator]) {

                        isLargest = false;

                      }

                    }

                    if (isLargest) {

                      data[col] += (max - min)/2;

                    }
                  }
                }
                if(TIFFWriteScanline(resultTif, data, row, 0) != -1){}
                else{
                  cout<<"ERROR WRITING RESULT LINE"<<endl;
                  exit(-1);
                }
              }
              else{
                cout<<"ERROR READING SCANLINE"<<endl;
                exit(-1);
              }
            }
            cout<<"pipeline visualization file has been created"<<endl;
            wFile.close();
            keyFile.close();
          }
          else{
            cout<<"FAILURE OPENING W_NMF file or key file"<<endl;
          }
          currentLine = "";
          TIFFClose(resultTif);
          _TIFFfree(buf);
          TIFFClose(tif);
        }
      }
      else{
        cout<<"COULD NOT OPEN "<<argv[1]<<endl;
        return 1;
      }
    }
    return 0;
}
