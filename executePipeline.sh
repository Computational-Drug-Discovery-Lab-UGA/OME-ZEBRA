#!/bin/bash
make clean
make -j9
mkdir /media/spacey-person/CDDL_Drive/NNMF_NOSVD
for dir in /media/spacey-person/CDDL_Drive/Registered/*/;
do
  justDirname=$(basename $dir)
  echo "$dir"
  mkdir /media/spacey-person/CDDL_Drive/NNMF_NOSVD/"${justDirname%%.*}"

  ./bin/ZEBRA.exe "${justDirname%%.*}" 512

  ./bin/NMF_GPU  /media/spacey-person/CDDL_Drive/NNMF_NOSVD/"${justDirname%%.*}"/NNMF.nmf  -k 3 -j 5 -t 20 -i 20000

  for((i = 0; i < 3; ++i));
  do
  ./bin/NNMF_VISUALIZE.exe "${justDirname%%.*}" 3 $i
  done
done
