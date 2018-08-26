#!/bin/bash
/bin/ZEBRA_NMF ../path/to/tifs/ 100 0
./bin/NMF_GPU  ../path/to/tifs/NNMF.nmf  -k 100 -j 10 -t 40  -i 20000
/bin/ZEBRA_NMF ../path/to/tifs/ 100 1

/bin/ZEBRA_NMF ../path/to/tifs/ 100 0
./bin/NMF_GPU  ../path/to/tifs/NNMF.nmf  -k 100 -j 10 -t 40  -i 20000
/bin/ZEBRA_NMF ../path/to/tifs/ 100 1

/bin/ZEBRA_NMF ../path/to/tifs/ 100 0
./bin/NMF_GPU  ../path/to/tifs/NNMF.nmf  -k 100 -j 10 -t 40  -i 20000
/bin/ZEBRA_NMF ../path/to/tifs/ 100 1
