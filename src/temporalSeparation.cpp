#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>

using namespace std;

int main(int argc, char *argv[]) {
  string tifName = argv[1];
  vector<vector<int>> tGroups;
  string nnmfFileLocation =  "data/out/" + tifName + "/NNMF.nmf";
  string hFileLocation = "data/out/" + tifName + "/NNMF.nmf_H.txt";
  ifstream height(hFileLocation);
  ifstream nnmf(nnmfFileLocation);
  string currentLine;
  if(height.is_open()){
    currentLine = "";
  }
  else{

  }
  height.close();
  


  currentLine = "";
  if(nnmf.is_open()){
    for(int i = 0; i < 10; ++i){
      string newNNMFLocation =  "data/out/" + tifName + "/NNMF_" + to_string(i) + ".nmf";
      ofstream newNNMF(newNNMFLocation);
      if(newNNMF.is_open()){

      }
      else{

      }
    }
  }
  else{

  }


  return 0;
}
