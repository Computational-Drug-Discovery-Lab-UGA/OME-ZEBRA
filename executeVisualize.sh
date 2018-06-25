#!/bin/bash

for dir in data/registeredOMEs/*
do
  justDirname=$(basename $dir)
  #visualize each
  for((tg = 0; tg < 10; ++tg));
  do
    for((i = 0; i < 3; ++i));
    do
      ./bin/NNMF_VISUALIZE.exe "${justDirname%%.*}" $tg 3 $i
    done
  done
done
