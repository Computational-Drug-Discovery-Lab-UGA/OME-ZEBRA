#include <iostream>
#include <fstream>
#include <string>

void multiplyMatrix(float* matrixA, float* matrixB, float* matrixC, int dimA,
     int dimCommon, int dimB);

int main(int argc, char const *argv[]) {

  int dimU = 3;
  int dimS = 3;
  int dimV = 3;

  std::ifstream sMatrixFile("data/sMatrix.txt");

  float* sMatrix = new float[dimS * dimS];
  std::string currentLine = "";
  int indexOfsMatrix = 0;
  int numSZero = 0;
  
  while (std::getline(sMatrixFile, currentLine)) {

      std::istringstream ss(currentLine);
      ss >> sMatrix[indexOfsMatrix];

      //cout<<vtMatrix[indexOfvtMatrix]<<endl;

      if(sMatrix[indexOfsMatrix] == 0.0f) ++numSZero;
      indexOfsMatrix++;

  }

  return 0;

}

void multiplyMatrix(float* matrixA, float* matrixB, float* matrixC, int dimA,
     int dimCommon, int dimB) {

  for (int i = 0; i < dimA; i++) {

    for (int j = 0; j < dimB; j++) {

      for (int k = 0; k < dimCommon; k++) {

        matrixC[dimB * i + j] = matrixC[dimB * i + j] + matrixA[dimCommon * i + k]
          * matrixB[dimB * k + j];

      }

    }

  }

}
