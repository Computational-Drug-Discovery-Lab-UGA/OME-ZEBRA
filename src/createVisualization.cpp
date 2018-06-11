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

    if(argc<2) {
      cout << "Usage: ./exe <file>";
      return 1;
    }
    else {
      int k = 2;
      string wFileLocation;
      string tifName = argv[1];
      string tifFile = "data/out/" + tifName + "/" + tifName + ".ome0000.tif";
      string nmfChecker = "data/out/" + tifName + "/NNMF.nmf";
      string keyFileLocation = "data/out/" + tifName + "/" + "key.csv";
      TIFF* tif = TIFFOpen(tifFile.c_str(), "r");
      if(argc == 3){
        istringstream argK(argv[2]);
        argK >> k;
        wFileLocation = "data/out/" + tifName + "/" + tifName + "_W.txt";
      }
      else if(argc == 4){//needs to have a better check
        string testCheck = argv[3];
        if(testCheck != "test"){cout<<"incorrect test flag"<<endl; exit(-1);}
        cout<<"test initiating"<<endl;
        istringstream argK(argv[2]);
        argK >> k;
        wFileLocation = "data/test.nmf_W.txt";
      }
      string fileName = "data/out/" + tifName + "/" + tifName + to_string(k) + "_RESULT.tif";
      cout<<wFileLocation<<endl;
      cout<<tifFile<<endl;
      cout<<keyFileLocation<<endl;
      cout<<fileName<<endl;
      cout<<"k = "<<k<<endl;

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
          int currentPixel = 0;
          ifstream wFile(wFileLocation);
          ifstream keyFile(keyFileLocation);
          ifstream nmfTest(nmfChecker);
          float* kArray = new float[k];
          string currentLine;
          string currentKey;
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
          if(wFile.is_open() && keyFile.is_open()){
            cout<<"key and W nmf file are open"<<endl;
            ofstream test("data/out/" + tifName + "/RESULT.csv");
            for (row = 0; row < height; ++row){
              if(TIFFReadScanline(tif, buf, row, 0) != -1){
                memcpy(data, buf, scanLineSize);
                getline(keyFile, currentKey);
                if(currentKey == "1"){
                  for(int col = 0; col < width; ++col){
                    getline(wFile, currentLine);
                    data[col] -= min;
                    stringstream ss;
                    ss<<currentLine;

                    for (int kIterator = 0; kIterator < k; kIterator++) {

                      ss>>kArray[kIterator];

                    }
                    bool isLargest = true;
                    for (int kIterator = 0; kIterator < k ; kIterator++) {

                      if (kArray[0] < kArray[kIterator]) {

                        isLargest = false;

                      }

                    }

                    if (isLargest) {

                      data[col] += (max - min)/2;

                    }
                    test<<data[col];
                    if(col != width - 1) test<<",";
                  }
                  test<<"\n"<<endl;
                }
                memcpy(buf, data, scanLineSize);
                if(TIFFWriteScanline(resultTif, data, row, 0) != 1){
                  cout<<"ERROR WRITING FIRST TIMEPOINT"<<endl;
                  exit(-1);
                }
                currentPixel++;
              }
              else{
                cout<<"ERROR READING SCANLINE"<<endl;
                exit(-1);
              }
            }
            test.close();
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
