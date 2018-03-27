CUDA_INSTALL_PATH := /usr/local/cuda

CXX := /usr/bin/g++-6
LINK := /usr/bin/g++-6
NVCC := nvcc

# Includes
INCLUDES = -I/usr/local/cuda/include -I/usr/local/include

# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_60,code=sm_60
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -g -std=c++11

LIB_CUDA := -L/usr/local/cuda-9.1/lib64 -lcudart
LIB_TIFF := -L/usr/local/lib -ltiff

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS = OME_ZEBRA.cu.o
OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = ZEBRA.exe
LINKLINE = ${LINK} -o ${BINDIR}/${TARGET} ${OBJS} ${LIB_CUDA} ${LIB_TIFF} ${INCLUDES}


.SUFFIXES: .cpp .cu .o

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} ${INCLUDES} -c $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${BINDIR}/${TARGET}: ${OBJS} Makefile
	${LINKLINE}

clean:
	rm -f bin/*.exe
	rm -f obj/*
	rm -f data/*.csv
	rm -f data/*TP1*
