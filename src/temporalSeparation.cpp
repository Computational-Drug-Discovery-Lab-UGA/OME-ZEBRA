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
  if(argc != 3){
    cout<<"Usage: ./exe <k> <numTimePoints>"<<endl;
    return 1;
  }
  string nnmfFileLocation =  "data/NNMF.nmf";
  string hFileLocation = "data/NNMF.nmf_H.txt";
  string keyFileLocation = "data/key.csv";
  int k = 2;
  int numPixels = 0;
  int numTimePoints = 512;
  istringstream argK(argv[1]);
  istringstream argTP(argv[2]);
  argK >> k;
  argTP >> numTimePoints;
  cout<<"k = "<<k<<" numTimePoints = "<<numTimePoints<<endl;
  float** heightMatrix = new float*[k];
  for(int i = 0; i < k; ++i){
    heightMatrix[i] = new float[numTimePoints];
  }
  ifstream height(hFileLocation);
  string currentLine;
  if(height.is_open()){
    cout<<"PARSING H FILE"<<endl;
    int row = 0;
    while(getline(height, currentLine)){
      istringstream ss(currentLine);
      for(int col = 0; col < numTimePoints; ++col){
        ss>>heightMatrix[row][col];
      }
      row++;
    }
    cout<<"DONE PARSING H FILE"<<endl;
    height.close();
  }
  else{
    cout<<"CANNOT OPEN "<<hFileLocation<<endl;
    return 1;
  }
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
    cout<<i<<" - "<<tGroups[i].size()<<endl;
  }
  vector<float*> nnmfMatrix;
  ifstream nnmf(nnmfFileLocation);
  if(nnmf.is_open()){
    cout<<"PARSING NNMF.nmf"<<endl;
    int row  = 0;
    while(getline(nnmf, currentLine)){
      istringstream ss(currentLine);
      float* temp = new float[numTimePoints];
      for(int col = 0; col < numTimePoints; ++col){
        ss >> temp[col];
      }
      nnmfMatrix.push_back(temp);
      row++;
    }
    numPixels = nnmfMatrix.size();
    nnmf.close();
    cout<<"DONE PARSING NNMF.nmf"<<endl;
  }
  else{
    cout<<"CANNOT OPEN "<<nnmfFileLocation<<endl;
    return 1;
  }
  currentLine = "";
  fstream key(keyFileLocation);
  vector<bool> keyValues;
  bool currentKey;
  int numKeys = 0;
  if(key.is_open()){
    while(getline(key, currentLine)){
      istringstream ss(currentLine);
      ss >> currentKey;
      keyValues.push_back(currentKey);
    }
    key.close();
    numKeys = keyValues.size();
  }
  else{
    cout<<"COULD NOT OPEN KEY"<<endl;
    return 1;
  }
  for(int row = 0; row < k; ++row){
    if(tGroups[row].size() == 0) continue;
    string newNNMFLocation =  "data/NNMF_" + to_string(row) + ".nmf";
    string newKeyLocation = "data/key_" + to_string(row) + ".csv";
    ofstream newNNMF(newNNMFLocation);
    ofstream newKey(newKeyLocation);
    cout<<"CREATING "<<newNNMFLocation<<endl;
    if(newNNMF.is_open() && newKey.is_open()){
      int pixel = 0;
      for(int i = 0; i < numKeys; ++i){
        if(!keyValues[i]){
          newKey << to_string(false);
          if(i != numKeys - 1)  newKey << "\n";
          continue;
        }
        bool allZero = true;
        ostringstream ss;
        allZero = true;
        for(int col = 0; col < tGroups[row].size(); ++col){
          if(allZero && nnmfMatrix[pixel][tGroups[row][col]] != 0.0f){
            allZero = false;
          }
          ss << to_string(nnmfMatrix[pixel][tGroups[row][col]]);
          if(col != tGroups[row].size() - 1){
             ss <<  " ";
          }
          else{
            ss << "\n";
          }
        }
        if(!allZero){
          newNNMF << ss.str();
        }
        newKey << to_string(!allZero);
        if(pixel != numKeys - 1){
          newKey << "\n";
        }
        pixel++;
      }
      cout<<"DONE WRITING "<<newNNMFLocation<<endl;
      newNNMF.close();
    }
    else{
      cout<<"CANNOT OPEN "<<newNNMFLocation<<endl;
      return 1;
    }
  }
  return 0;
}
