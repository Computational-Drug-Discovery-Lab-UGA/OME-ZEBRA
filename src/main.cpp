#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include "tiffio.h"


using namespace std;

void printArray(uint16 * array, uint16 width);
vector<uint16*> extractMartrices(TIFF* tif);

int main(int argc, char *argv[]) {

    vector<vector<uint16*>> fullTiffVector;
    if(argc!=2) {
      cout << "Usage: ./exe <file>";
      return 1;
    }
    else {
      TIFF* tif = TIFFOpen(argv[1], "r");
      cout<<endl<<argv[1]<<" IS OPENED\n"<<endl;
      if (tif) {
          int dircount = 0;
          do {
            fullTiffVector.push_back(extractMartrices(tif));
            dircount++;
          }
          while (TIFFReadDirectory(tif));
          printf("%d directories in %s\n", dircount, argv[1]);

          //to save numColumns
          uint32 numColumns;
          TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &numColumns);

          TIFFClose(tif);

          //access matrix with fullTiffVector = timePoints
          printf("Total TimePoints = %d\nTotal Rows = %d\nTotal Columns = %d\n",fullTiffVector.size(), fullTiffVector[0].size(), numColumns);




      }
      else{
        cout<<"COULD NOT OPEN"<<argv[1];
        return 1;
      }
    }
    return 0;
}


void printArray(uint16 * array, uint16 width){
    uint32 i;
    for (i=0;i<width;i++){
      printf("%u ", array[i]);
    }
    cout<<endl;
}
vector<uint16*> extractMartrices(TIFF* tif){

  uint32 height,width;
  tdata_t buf;
  uint32 row;
  uint32 config;
  vector<uint16*> currentPlane;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  buf = _TIFFmalloc(TIFFScanlineSize(tif));

  uint16* data;
  //printf("Height,Width = %u,%u -> scanLineSize = %d bytes\n", height, width,TIFFScanlineSize(tif));
  for (row = 0; row < height; row++){
    TIFFReadScanline(tif, buf, row);
    data=(uint16*)buf;
    currentPlane.push_back(data);
    //printArray(data,width);//make sure you have a big screen
  }
  //cout<<endl<<endl;//if you are using the printArray method
  _TIFFfree(buf);
  return currentPlane;
}
