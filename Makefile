CUDA_INSTALL_PATH := /usr/local/cuda

CXX := /usr/bin/g++
LINK := nvcc
NVCC := nvcc
FORT := gfortran


# Includes
INCLUDES = -I/usr/local/cuda/include -I/usr/local/include
# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_60,code=sm_60 -Iinclude
CXXFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -Iinclude

LIB_CUDA := -L/usr/local/cuda/lib64 -lcudart
LIB_TIFF := -L/usr/local/lib -ltiff

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS = cuda_zebra.cu.o
_OBJS += io_util.cu.o
_OBJS += OME_ZEBRA.cu.o
OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = ZEBRA_NMF
LINKLINE = ${LINK} -o ${BINDIR}/${TARGET} ${OBJS} ${LIB_CUDA} ${LIB_TIFF} ${INCLUDES}


.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET}

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} ${INCLUDES} -c $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${BINDIR}/${TARGET}: ${OBJS} Makefile
	${LINKLINE}

clean:
	rm -f bin/ZEBRA_NMF
	rm -f obj/*

config:
	mkdir obj
	mkdir bin
	mkdir data
