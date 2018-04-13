#!/bin/bash

for dir in data/registeredOMEs/*
do
  mkdir data/out
  justDirname=$(basename $dir)
  ./bin/ZEBRA.exe "${justDirname%%.*}" 512
  ./bin/NMF_GPU  data/NNMF.nmf  -k 2  -j 10  -t 40  -i 20000


  mkdir data/out/"${justDirname%%.*}"
  mv data/NNMF.nmf_H.txt data/out/"${justDirname%%.*}"/"${justDirname%%.*}"_H.txt
  mv data/NNMF.nmf_W.txt data/out/"${justDirname%%.*}"/"${justDirname%%.*}"_W.txt
  cp data/registeredOMEs/"${justDirname%%.*}"/"${justDirname%%.*}".ome0000.tif data/out/"${justDirname%%.*}"/
  mv data/key.csv data/out/"${justDirname%%.*}"/key.csv
  mv data/NNMF.nmf data/out/"${justDirname%%.*}"/NNMF.nmf

  ./bin/NNMF_VISUALIZE.exe "${justDirname%%.*}" 2
  mv data/RESULT.csv data/out/"${justDirname%%.*}"/RESULT.csv


  rm data/NNMF.csv
done
