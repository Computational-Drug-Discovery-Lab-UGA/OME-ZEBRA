CUDA_INSTALL_PATH := /usr/local/apps/cuda/9.0.176_384.81
CXX := /usr/bin/g++
LINK := nvcc
NVCC := nvcc

# Includes
INCLUDES = -I/usr/local/apps/cuda/9.0.176_384.81/include -I/usr/local/include
INCLUDES += -I/usr/include/python3.5/
# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_60,code=sm_60 -Iinclude
CXXFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -Iinclude

LIB_CUDA := -L/usr/local/apps/cuda/9.0.176_384.81/lib64 -lcudart
LIB_TIFF := -L/usr/local/lib -ltiff

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS = cuda_zebra.cu.o
_OBJS += io_util.cu.o
_OBJS += OME_ZEBRA.cu.o
OBJS = ${patsubst %, ${OBJDIR}/%, ${_OBJS}}

TARGET = ZEBRA_NMF
LINKLINE = ${LINK} -gencode=arch=compute_60,code=sm_60 ${OBJS} \
${LIB_CUDA} ${LIB_TIFF} ${INCLUDES} -o ${BINDIR}/${TARGET}


.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET}

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} ${INCLUDES} -dc $< -o $@

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
