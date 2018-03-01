#include <iostream>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <vector>
#include <algorithm>

//this will most likely just get droped on top of main

using namespace std;
int* flattenMatrix(int** matrix, int cols, int rows){
  int* flat = new int[rows*cols];
  int currentPixel = 0;
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      flat[currentPixel] = matrix[r][c];
      ++currentPixel;
    }
  }
  cout<<"Matrix is flattened."<<endl;
  return flat;
}
int** transposeMatrix(int transpose, int** matrix, int cols, int rows){
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c){
      matrix[r][c] += transpose;
    }
  }
  return matrix;
}
int main(){

  int numTimePoints = 25;
  int rows  = 25;
  int columns = 25;
  int** testMatrix = new int*[rows];
  for(int i = 0; i < rows; ++i){
    testMatrix[i] = new int[columns];
    for(int c = 0; c < columns; ++c){
      testMatrix[i][c] = c;
    }
  }

  cout<<"Done filling test array"<<endl;
  vector<int*> timepoints;
  for(int i = 0; i < numTimePoints; ++i){

    int* testTimePoint = flattenMatrix(transposeMatrix(1, testMatrix, columns, rows), columns, rows);
    timepoints.push_back(testTimePoint);
  }
  cout<<"Done filling timepoint vector"<<endl;
  cout<<"Printing last timepoint:"<<endl;
  for(int i = 0; i < rows*columns; ++i){
    if(i%columns == 0){
      cout<<endl;
    }
    cout<<timepoints[numTimePoints - 1][i]<<" ";
  }
  cout<<endl;

  return 0;
}
