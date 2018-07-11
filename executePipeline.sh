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

  ./bin/NMF_GPU  /media/spacey-person/CDDL_Drive/NNMF_NOSVD/"${justDirname%%.*}"/NNMF.nmf  -k 10 -j 10 -t 40 -i 20000

  ./bin/NNMF_VISUALIZE.exe "${justDirname%%.*}" 10 
done
