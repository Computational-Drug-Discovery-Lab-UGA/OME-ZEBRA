#!/bin/bash
make clean
make
for((i = 2; i <= 10; ++i));
do
  for dir in data/registeredOMEs/*
  do
    mkdir data/out
    justDirname=$(basename $dir)
    ./bin/ZEBRA.exe "${justDirname%%.*}" 512
    ./bin/NMF_GPU  data/NNMF.nmf  -k $i -j 5  -t 20  -i 20000


    mkdir data/out/"${justDirname%%.*}"
    mv data/NNMF.nmf_H.txt data/out/"${justDirname%%.*}"/"${justDirname%%.*}"_H.txt
    mv data/NNMF.nmf_W.txt data/out/"${justDirname%%.*}"/"${justDirname%%.*}"_W.txt
    cp data/registeredOMEs/"${justDirname%%.*}"/"${justDirname%%.*}".ome0000.tif data/out/"${justDirname%%.*}"/
    mv data/key.csv data/out/"${justDirname%%.*}"/key.csv
    mv data/NNMF.nmf data/out/"${justDirname%%.*}"/NNMF.nmf

    for((ii = 0; ii < i; ++ii));
    do
    ./bin/NNMF_VISUALIZE.exe "${justDirname%%.*}" $i $ii
    done
  done
done
