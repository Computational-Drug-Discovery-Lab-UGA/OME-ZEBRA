CUDA_INSTALL_PATH := /usr/local/cuda

CXX := /usr/bin/g++-6
LINK := /usr/bin/g++-6
NVCC := nvcc

# Includes
INCLUDES = -I/usr/local/cuda/include -I/usr/local/include

# Common flags
COMMONFLAGS += ${INCLUDES}
NVCCFLAGS += ${COMMONFLAGS}
NVCCFLAGS += -std=c++11 -gencode=arch=compute_60,code=sm_60 -Iinclude -lcublas
CXXFLAGS += ${COMMONFLAGS}
CXXFLAGS += -Wall -g -std=c++11 -Iinclude -lcublas

LIB_CUDA := -L/usr/local/cuda/lib64 -lcudart -lcublas
LIB_TIFF := -L/usr/local/lib -ltiff

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

_OBJS1 = OME_ZEBRA.cu.o
OBJS1 = ${patsubst %, ${OBJDIR}/%, ${_OBJS1}}

_OBJS2 = createVisualization.cpp.o
OBJS2 = ${patsubst %, ${OBJDIR}/%, ${_OBJS2}}

_OBJS3 = temporalSeparation.cpp.o
OBJS3 = ${patsubst %, ${OBJDIR}/%, ${_OBJS3}}

TARGET1 = ZEBRA.exe
TARGET2 = NNMF_VISUALIZE.exe
TARGET3 = TEMPORAL_SEPARATION.exe
LINKLINE1 = ${LINK} -o ${BINDIR}/${TARGET1} ${OBJS1} ${LIB_CUDA} ${LIB_TIFF} ${INCLUDES}
LINKLINE2 = ${LINK} -o ${BINDIR}/${TARGET2} ${OBJS2} ${LIB_TIFF} ${INCLUDES}
LINKLINE3 = ${LINK} -o ${BINDIR}/${TARGET3} ${OBJS3} ${INCLUDES}


.SUFFIXES: .cpp .cu .o

all: ${BINDIR}/${TARGET1} ${BINDIR}/${TARGET2} ${BINDIR}/${TARGET3}

${OBJDIR}/%.cu.o: ${SRCDIR}/%.cu
	${NVCC} ${NVCCFLAGS} ${INCLUDES} -c $< -o $@

${OBJDIR}/%.cpp.o: ${SRCDIR}/%.cpp
	${CXX} ${CXXFLAGS} ${INCLUDES} -c $< -o $@

${BINDIR}/${TARGET1}: ${OBJS1} Makefile
	${LINKLINE1}

${BINDIR}/${TARGET2}: ${OBJS2} Makefile
	${LINKLINE2}

${BINDIR}/${TARGET3}: ${OBJS3} Makefile
	${LINKLINE3}

clean:
	rm -f bin/*.exe
	rm -f obj/*
	rm -f -r data/out/
	rm -f data/registeredOMEs/*TP*.tif
	rm -f *.o
	rm -f data/*.csv
	rm -f data/*.nmf

config:
	mkdir obj
	mkdir bin
	mkdir data
