#!/bin/bash
make clean
make -j9
mkdir data/out

for dir in data/registeredOMEs/*
do
  justDirname=$(basename $dir)
  mkdir data/out/"${justDirname%%.*}"

  ./bin/ZEBRA.exe "${justDirname%%.*}" 512
  ./bin/NMF_GPU  data/NNMF.nmf  -k 10 -j 5 -t 40  -i 20000

  #separate TIMEPOINTS
  ./bin/TEMPORAL_SEPARATION.exe 10 512 524288

  #NNMF on each timepoint
  for((i = 0; i < 10; ++i));
  do
    ./bin/NMF_GPU  data/NNMF_"$i".nmf  -k 3 -j 5 -t 40  -i 20000
  done

  mv data/*.txt data/out/"${justDirname%%.*}"/
  mv data/*.nmf data/out/"${justDirname%%.*}"/
  cp data/registeredOMEs/"${justDirname%%.*}"/"${justDirname%%.*}".ome0000.tif data/out/"${justDirname%%.*}"/
  mv data/key.csv data/out/"${justDirname%%.*}"/key.csv

  #visualize each
  for((tg = 0; tg < 10; ++tg));
  do
    for((i = 0; i < 3; ++i));
    do
      ./bin/NNMF_VISUALIZE.exe "${justDirname%%.*}" $tg 3 $i
    done
  done
done
