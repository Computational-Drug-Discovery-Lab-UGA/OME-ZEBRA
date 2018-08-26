#!/bin/bash

ml CUDA/9.0.176-GCC-6.4.0-2.28
ml LibTIFF/4.0.9-GCCcore-6.4.0
make clean
make

./bin/ZEBRA_NMF "../node_zebrafish_test/registered20161028/registered gad1b+,- heterozygotic_2/registeredfish2_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/" 100 0
./bin/NMF_GPU   "../node_zebrafish_test/registered20161028/registered gad1b+,- heterozygotic_2/registeredfish2_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/NNMF.txt"  -k 100 -j 10 -t 40  -i 20000
./bin/ZEBRA_NMF "../node_zebrafish_test/registered20161028/registered gad1b+,- heterozygotic_2/registeredfish2_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/" 100 1

./bin/ZEBRA_NMF "../node_zebrafish_test/registered20161028/registered gad1b+,- heterozygotic_3/registeredfish3_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/" 100 0
./bin/NMF_GPU   "../node_zebrafish_test/registered20161028/registered gad1b+,- heterozygotic_3/registeredfish3_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/NNMF.txt"  -k 100 -j 10 -t 40  -i 20000
./bin/ZEBRA_NMF "../node_zebrafish_test/registered20161028/registered gad1b+,- heterozygotic_3/registeredfish3_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/" 100 1

./bin/ZEBRA_NMF "../node_zebrafish_test/registered20161028/registered gad1b wild type/registeredfish1_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/" 100 0
./bin/NMF_GPU   "../node_zebrafish_test/registered20161028/registered gad1b wild type/registeredfish1_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/NNMF.txt"  -k 100 -j 10 -t 40  -i 20000
./bin/ZEBRA_NMF "../node_zebrafish_test/registered20161028/registered gad1b wild type/registeredfish1_5d_gcamp5g_gad1b_rfp_WTorHet_25fps_5mw_488(1)/" 100 1
