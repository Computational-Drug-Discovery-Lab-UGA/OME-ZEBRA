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
  if(argc != 5){
    cout<<"Usage: ./exe <fileName> <k> <numTimePoints> <numPixels>"<<endl;
    return 1;
  }
  string tifName = argv[1];
  string nnmfFileLocation =  "data/out/" + tifName + "/NNMF.nmf";
  string hFileLocation = "data/out/" + tifName + "/NNMF.nmf_H.txt";
  int k = 2;
  int numTimePoints = 512;
  int numPixels = 0;
  istringstream argK(argv[2]);
  istringstream argTP(argv[3]);
  istringstream argP(argv[4]);
  argK >> k;
  argTP >> numTimePoints;
  argP >> numPixels;
  float** heightMatrix = new float*[k];
  for(int i = 0; i < k; ++i){
    heightMatrix[i] = new float[numTimePoints];
  }
  ifstream height(hFileLocation);
  string currentLine;
  if(height.is_open()){
    int row = 0;
    while(getline(height, currentLine)){
      istringstream(currentLine);
      for(int col = 0; col < numTimePoints; ++col){
        currentLine>>heightMatrix[row][col];
      }
      ++row;
    }
  }
  else{
    cout<<"CANNOT OPEN "<<hFileLocation<<endl;
    return 1;
  }
  height.close();
  vector<vector<int>> tGroups;
  for(int col = 0; col < numTimePoints; ++col){
    int greatestIndex = 0;
    float greatest = 0;
    int row = 0;
    for(; row < k; ++row){
      if(greatest < heightMatrix[row][col]){
        greatestIndex = row;
      }
    }
    tGroups[row].push_back(col);
  }
  float** nnmfMatrix = new float*[numPixels];
  for(int i = 0; i < numPixels; ++i){
    nnmfMatrix[i] = new float[numTimePoints];
  }
  ifstream nnmf(nnmfFileLocation);
  if(nnmf.is_open()){
    int row = 0;
    while(getline(nnmf, currentLine)){
      istringstream(currentLine);
      for(int col = 0; col < numTimePoints; ++col){
        currentLine>>nnmfMatrix[row][col];
      }
      ++row;
    }
  }
  else{
    cout<<"CANNOT OPEN "<<nnmfFileLocation<<endl;
    return 1;
  }
  nnmf.close();
  currentLine = "";
  for(int row = 0; row < k; ++row){
    if(tGroups[row].size() == 0) continue;
    string newNNMFLocation =  "data/out/" + tifName + "/NNMF_" + to_string(row) + ".nmf";
    ofstream newNNMF(newNNMFLocation);
    if(newNNMF.is_open()){
      for(int pixel = 0; pixel < numPixels; ++pixel){
        ostringstream pixelString;
        for(int col = 0; col < tGroups[row].size(); ++col){
          pixelString << nnmfMatrix[pixel][tGroups[row][col]];
        }
        newNNMF << pixelString.str();
      }
    }
    else{
      cout<<"CANNOT OPEN "<<newNNMFLocation<<endl;
      return 1;
    }
    newNNMF.close();
  }
  return 0;
}
