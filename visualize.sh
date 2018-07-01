#!/bin/bash
make clean
make -j9
mkdir /media/spacey-person/CDDL_Drive/NNMF_NOSVD
for dir in /media/spacey-person/CDDL_Drive/Registered/*/;
do
  justDirname=$(basename $dir)
  echo "$dir"

  for((i = 0; i < 3; ++i));
  do
  ./bin/NNMF_VISUALIZE.exe "${justDirname%%.*}" 3 $i
  done
done
