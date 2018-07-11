#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

void multiplyMatrix(float* matrixA, float* matrixB, float* matrixC, int dimA,
     int dimCommon, int dimB);
void applyScalar(float* targetMatrix, float* newMatrix, float* numerator, float* denominator, int numRows, int numCols);
float calculateLoss(float* originalMatrix, float* testMatrix, int numRows, int numCols);
void updateHeight(float* heightMatrix, float* widthMatrix, float* uMatrix, float* sMatrix,
  float* vtMatrix, float* newHeightMatrix, int numPixels, int numTime, int numSingularValues);
void updateWidth(float* heightMatrix, float* widthMatrix,
    float* uMatrix, float* sMatrix, float* vtMatrix, float* newWidthMatrix,
    int numPixels, int numTime, int numSingularValues);
void NMF(float* heightMatrix, float* widthMatrix, float* uMatrix,
  float* sMatrix, float* vtMatrix, float* originalMatrix, int numPixels, int numTime,
  int numSingularValues, float targetLoss);

int main(int argc, char const *argv[]) {

  int dimU = 500;
  int dimS = 200;
  int dimV = 200;

  std::cout << "Inputting sMatrix" << '\n';

  std::ifstream sMatrixFile("data/sMatrix.txt");

  float* sMatrix = new float[dimS * dimS];
  std::string currentLine = "";
  int indexOfsMatrix = 0;

  while (std::getline(sMatrixFile, currentLine)) {

      std::istringstream ss(currentLine);
      ss >> sMatrix[indexOfsMatrix];
      std::cout << sMatrix[indexOfsMatrix] << '\n';
      indexOfsMatrix++;

  }

  std::cout << "Inputting uMatrix" << '\n';

  std::ifstream uMatrixFile("data/uMatrix.txt");

  float* uMatrix = new float[dimU * dimS];
  int indexOfuMatrix = 0;

  while (std::getline(uMatrixFile, currentLine)) {

      std::istringstream ss(currentLine);
      ss >> uMatrix[indexOfuMatrix];
      std::cout << uMatrix[indexOfuMatrix] << '\n';
      indexOfuMatrix++;

  }

  std::cout << "Inputting vtMatrix" << '\n';

  std::ifstream vtMatrixFile("data/vtMatrix.txt");

  float* vtMatrix = new float[dimU * dimS];
  int indexOfvtMatrix = 0;

  while (std::getline(vtMatrixFile, currentLine)) {

      std::istringstream ss(currentLine);
      ss >> vtMatrix[indexOfvtMatrix];
      std::cout << vtMatrix[indexOfvtMatrix] << '\n';
      indexOfvtMatrix++;

  }

  std::cout << "Inputting originalMatrix" << '\n';

  std::ifstream originalMatrixFile("data/originalMatrix.txt");

  float* originalMatrix = new float[dimU * dimS];
  int indexOforiginalMatrix = 0;

  while (std::getline(originalMatrixFile, currentLine)) {

      std::istringstream ss(currentLine);
      ss >> originalMatrix[indexOforiginalMatrix];
      std::cout << originalMatrix[indexOforiginalMatrix] << '\n';
      indexOforiginalMatrix++;

  }

  float* heightMatrix = new float[dimS * dimV];

  for(int i = 0; i < dimS*dimV; i++) {

    heightMatrix[i] = 1.0f;

  }

  float* widthMatrix = new float[dimU * dimS];

  for(int i = 0; i < dimU*dimV; i++) {

    widthMatrix[i] = 1.0f;

  }

  float targetLoss = 0.9;

  NMF(heightMatrix, widthMatrix, uMatrix, sMatrix, vtMatrix, originalMatrix, dimU, dimV,
    dimS, targetLoss);

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

void applyScalar(float* targetMatrix, float* newMatrix, float* numerator, float* denominator, int numRows, int numCols) {

  for (int i = 0; i < numRows; i++) {

    for (int j = 0; j < numCols; j++) {

      int index = numCols * i + j;

      newMatrix[index] = targetMatrix[index] * (numerator[index] / denominator[index]);

    }

  }

}

float calculateLoss(float* originalMatrix, float* testMatrix, int numRows, int numCols) {

  float currentLoss = 0;

  for (int i = 0; i < numRows; i++) {

    for (int j = 0; j < numCols; j++) {

      int index = numCols * i + j;

      currentLoss = currentLoss + std::abs(originalMatrix[index] - testMatrix[index]);

    }

  }

  return currentLoss;

}

void updateHeight(float* heightMatrix, float* widthMatrix,
                  float* uMatrix, float* sMatrix, float* vtMatrix, float* newHeightMatrix,
                  int numPixels, int numTime, int numSingularValues) {

  float* widthMatrixTransposed = new float[numPixels * numSingularValues];

  for (long i = 0; i < numPixels; i++) {

    for (long j = 0; j < numSingularValues; j++) {

      widthMatrixTransposed[j * numPixels + i] = widthMatrix[i * numSingularValues + j];

    }

  }

  float* tempSquareMatrix = new float[numSingularValues * numSingularValues];

  multiplyMatrix(widthMatrixTransposed, uMatrix, tempSquareMatrix,
    numSingularValues, numPixels, numSingularValues);

  float* tempSquareMatrix2 = new float[numSingularValues * numSingularValues];

  multiplyMatrix(tempSquareMatrix, sMatrix, tempSquareMatrix2, numSingularValues, numSingularValues, numSingularValues);

  float* numerator = new float[numSingularValues * numTime];

  multiplyMatrix(tempSquareMatrix2, vtMatrix, numerator, numSingularValues, numSingularValues, numTime);

  multiplyMatrix(widthMatrixTransposed, widthMatrix, tempSquareMatrix,
    numSingularValues, numPixels, numSingularValues);

  float* denominator = new float[numSingularValues * numTime];

  multiplyMatrix(tempSquareMatrix, heightMatrix,
    denominator, numSingularValues, numSingularValues, numTime);

  applyScalar(heightMatrix, newHeightMatrix, numerator, denominator, numSingularValues, numTime);

}

void updateWidth(float* heightMatrix, float* widthMatrix,
                 float* uMatrix, float* sMatrix, float* vtMatrix,
                 float* newWidthMatrix, int numPixels, int numTime,
                 int numSingularValues) {

    float* heightMatrixTransposed = new float[numSingularValues * numTime];

    for (long i = 0; i < numSingularValues; i++) {

      for (long j = 0; j < numTime; j++) {

        heightMatrixTransposed[j * numSingularValues + i] = heightMatrix[i * numTime + j];

      }

    }



    float* tempSquareMatrix = new float[numSingularValues * numSingularValues];

    multiplyMatrix(vtMatrix, heightMatrixTransposed, tempSquareMatrix,
       numSingularValues, numTime, numSingularValues);



    float* tempSquareMatrix2 = new float[numSingularValues * numSingularValues];

    multiplyMatrix(sMatrix, tempSquareMatrix, tempSquareMatrix2, numSingularValues,
        numSingularValues, numSingularValues);



    float* numerator = new float[numPixels * numSingularValues];

    multiplyMatrix(uMatrix, tempSquareMatrix2, numerator, numPixels,
      numSingularValues, numSingularValues);



    multiplyMatrix(heightMatrix, heightMatrixTransposed, tempSquareMatrix,
      numSingularValues, numTime, numSingularValues);



    float* denominator = new float[numPixels * numSingularValues];

    multiplyMatrix(widthMatrix, tempSquareMatrix, denominator, numPixels,
      numSingularValues, numSingularValues);



    applyScalar(widthMatrix, newWidthMatrix, numerator, denominator, numPixels,
      numSingularValues);

}

void NMF(float* heightMatrix, float* widthMatrix, float* uMatrix,
  float* sMatrix, float* vtMatrix, float* originalMatrix, int numPixels, int numTime,
  int numSingularValues, float targetLoss) {

  float* newWidthMatrix = new float[numPixels * numSingularValues];
  float* newHeightMatrix = new float[numSingularValues * numTime];

  float loss = 10000.0;

  while(loss > targetLoss) {

    updateHeight(heightMatrix, widthMatrix, uMatrix, sMatrix, vtMatrix,
      newHeightMatrix, numPixels, numTime, numSingularValues);

    updateWidth(heightMatrix, widthMatrix, uMatrix, sMatrix, vtMatrix,
      newWidthMatrix, numPixels, numTime, numSingularValues);

    float* testMatrix = new float[numPixels * numTime];

    multiplyMatrix(newWidthMatrix, newHeightMatrix, testMatrix,
      numPixels, numSingularValues, numTime);

    loss = calculateLoss(originalMatrix, testMatrix, numPixels, numTime);

    for (int i = 0; i < numSingularValues * numTime; i++) {

      heightMatrix[i] = newHeightMatrix[i];

    }

    for (int i = 0; i < numPixels * numSingularValues; i++) {

      widthMatrix[i] = newWidthMatrix[i];

    }

    std::cout << "Loss is: " << loss << '\n';

  }

}
