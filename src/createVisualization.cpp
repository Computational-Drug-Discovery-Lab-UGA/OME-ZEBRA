#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include "tiffio.h"
#include <fstream>

using namespace std;


int main(int argc, char *argv[]) {

    if(argc!=2) {
      cout << "Usage: ./exe <file>";
      return 1;
    }
    else {

      TIFF* tif = TIFFOpen(argv[1], "w");
      cout<<endl<<argv[1]<<" IS OPENED\n"<<endl;
      string fileName = argv[1];
      int dircount = 0;
      if (tif) {

      }
      else{
        cout<<"COULD NOT OPEN"<<argv[1];
        return 1;
      }
    }
    return 0;
}
