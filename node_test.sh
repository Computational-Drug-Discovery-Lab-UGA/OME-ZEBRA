#!/bin/bash
/bin/ZEBRA_NMF ../ 100
./bin/NMF_GPU  data/NNMF.nmf  -k 100 -j 5  -t 20  -i 20000
/bin/ZEBRA_NMF ../ 100
