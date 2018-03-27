#!/bin/bash

for file in registeredOMEs/*.ome.tif
do
  bin/./ZEBRA.exe "$file"
  bin/./NMF_GPU  data/new.csv  -k 2  -j 10  -t 40  -i 20000

  justFilename=$(basename $file)

  mkdir output/"${justFilename%%.*}"
  mv data/new.csv_H.txt output/"${justFilename%%.*}"/"${justFilename%%.*}"_H.txt
  mv data/new.csv_W.txt output/"${justFilename%%.*}"/"${justFilename%%.*}"_W.txt
  mv data/key.csv output/"${justFilename%%.*}"/"${justFilename%%.*}"_key.csv

  rm data/new.csv
done
