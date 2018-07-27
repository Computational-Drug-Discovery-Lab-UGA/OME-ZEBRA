#ifndef MAGMA_SVD_H
#define MAGMA_SVD_H

#include <stdio.h>
#include <cuda.h>
#include "magma_v2.h"
#include "magma_lapack.h"
#include <iostream>
#include <fstream>

void performSVD(long mValue, long nValue, float* originalMatrix, float* sMatrix; float* uMatrix, float* vtMatrix);

#endif
