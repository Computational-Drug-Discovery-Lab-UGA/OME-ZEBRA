#!/bin/bash

for file in data/registeredOMEs/*.ome.tif
do
  mkdir data/out
  ./bin/ZEBRA.exe "$file"
  ./bin/NMF_GPU  data/NNMF.csv  -k 2  -j 10  -t 40  -i 20000

  justFilename=$(basename $file)

  mkdir data/out/"${justFilename%%.*}"
  mv data/NNMF.csv_H.txt data/out/"${justFilename%%.*}"/"${justFilename%%.*}"_H.txt
  mv data/NNMF.csv_W.txt data/out/"${justFilename%%.*}"/"${justFilename%%.*}"_W.txt
  mv data/registeredOMEs/"${justFilename%%.*}"_TP1.tif data/out/"${justFilename%%.*}"/"${justFilename%%.*}"_TP1.tif
  mv data/TP1.csv data/out/"${justFilename%%.*}"/TP1.csv
  mv data/key.csv data/out/"${justFilename%%.*}"/key.csv

  ./bin/NNMF_VISUALIZE.exe "${justFilename%%.*}"
  mv data/RESULT.csv data/out/"${justFilename%%.*}"/RESULT.csv


  rm data/NNMF.csv
done
