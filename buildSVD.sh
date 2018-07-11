g++ -O3 -fopenmp -std=c++11 -DHAVE_CUBLAS -DADD_ -I/usr/local/cuda/include -I/usr/local/magma/include -c -o svd.o svd.cpp

g++ -fopenmp -o svd svd.o -L/usr/local/magma/lib -lm -lmagma -L/usr/local/cuda/lib64 -L/usr/lib -L/opt/openblas/lib -lopenblas -lcublas -lcudart

g++ -c nnmf.cpp

g++ -O3 -o nnmf nnmf.o
