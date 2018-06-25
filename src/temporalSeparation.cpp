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
  if(argc != 4){
    cout<<"Usage: ./exe <k> <numTimePoints> <numPixels>"<<endl;
    return 1;
  }
  string nnmfFileLocation =  "data/NNMF.nmf";
  string hFileLocation = "data/NNMF.nmf_H.txt";
  int k = 2;
  int numTimePoints = 512;
  int numPixels = 0;
  istringstream argK(argv[1]);
  istringstream argTP(argv[2]);
  istringstream argP(argv[3]);
  argK >> k;
  argTP >> numTimePoints;
  argP >> numPixels;
  cout<<"k = "<<k<<" numTimePoints = "<<numTimePoints<<" numPixels = "<<numPixels<<endl;
  float** heightMatrix = new float*[k];
  for(int i = 0; i < k; ++i){
    heightMatrix[i] = new float[numTimePoints];
  }
  ifstream height(hFileLocation);
  string currentLine;
  if(height.is_open()){
    cout<<"PARSING H FILE"<<endl;
    getline(height, currentLine);
    for(int row = 0; row < k; ++row){
      istringstream(currentLine);
      for(int col = 0; col < numTimePoints; ++col){
        currentLine>>heightMatrix[row][col];
      }
    }
    cout<<"DONE PARSING H FILE"<<endl;
  }
  else{
    cout<<"CANNOT OPEN "<<hFileLocation<<endl;
    return 1;
  }
  height.close();
  vector<vector<int>> tGroups;
  for(int i = 0; i < k; ++i){
    vector<int> kVector;
    tGroups.push_back(kVector);
  }
  for(int col = 0; col < numTimePoints; ++col){
    int greatestIndex = 0;
    float greatest = 0;
    for(int row = 0; row < k; ++row){
      if(greatest < heightMatrix[row][col]){
        greatestIndex = row;
        greatest = heightMatrix[row][col];
      }
    }
    tGroups[greatestIndex].push_back(col);
  }
  for(int i = 0; i < k; ++i){
    cout<<i<<" - "<<tGroups[i].size();
  }
  float** nnmfMatrix = new float*[numPixels];
  for(int i = 0; i < numPixels; ++i){
    nnmfMatrix[i] = new float[numTimePoints];
  }
  ifstream nnmf(nnmfFileLocation);
  if(nnmf.is_open()){
    cout<<"PARSING NNMF.nmf"<<endl;
    getline(nnmf, currentLine);
    istringstream ss(currentLine);
    for(int row = 0; row < numPixels; ++row){
      for(int col = 0; col < numTimePoints; ++col){
        ss >> nnmfMatrix[row][col];
      }
    }
    cout<<"DONE PARSING NNMF.nmf"<<endl;
  }
  else{
    cout<<"CANNOT OPEN "<<nnmfFileLocation<<endl;
    return 1;
  }
  nnmf.close();
  currentLine = "";

  for(int row = 0; row < k; ++row){
    if(tGroups[row].size() == 0) continue;
    string newNNMFLocation =  "data/NNMF_" + to_string(row) + ".nmf";
    ofstream newNNMF(newNNMFLocation);
    if(newNNMF.is_open()){
      cout<<"CREATING "<<newNNMFLocation<<endl;
      for(int pixel = 0; pixel < numPixels; ++pixel){
        ostringstream pixelString;
        for(int col = 0; col < tGroups[row].size(); ++col){
          pixelString << nnmfMatrix[pixel][tGroups[row][col]];
        }
        newNNMF << pixelString.str() + "\n";
      }
      cout<<"DONE WRITING "<<newNNMFLocation<<endl;
    }
    else{
      cout<<"CANNOT OPEN "<<newNNMFLocation<<endl;
      return 1;
    }
    newNNMF.close();
  }
  return 0;
}
