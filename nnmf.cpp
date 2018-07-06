#include <iostream>
#include <fstream>

void multiplyMatrix(float* matrixA, float* matrixB, float* matrixC, int dimA,
     int dimCommon, int dimB);

int main(int argc, char const *argv[]) {

  

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
