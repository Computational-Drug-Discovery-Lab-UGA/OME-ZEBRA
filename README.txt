nvcc -std=c++11 -c MatrixOperations.o MatrixOperations.cu
nvcc -std=c++11 -c Utilities.cu Utilities.o
nvcc -std=c++11 -o SVD.exe MatrixOperations.o Utilities.o
