necessary directories
bin,obj,src

FILE NAMING CONVENTION
  basename0000.tif
  (0000 is where timepoint is noted)

Cuda/c++/c dependencies
libtiff,cuda-9
-get Python.h flags for linking and compiling by doing This
  $ /path/to/bin/python3.n-config --cflags
  $ /path/to/bin/python3.n-config --ldflags

python dependencies
python-3.5,tensorflow-1.10,numpy,pandas

USAGE: (from */OME-ZEBRA/ directory)
  ./bin/ZEBRA_NMF /path/to/image/sequence/folder <args>

  Arguments:
    <specifier> <value> - description
    -k 2 (# of factors)
    -i 1000 (max iterations)
    -l .1 (learning rate)
    -t 1e-8 (prev cost - cost < threshold = convergence)
    -s 9.0 (sigmoid tuning, 9.0 -> 1/(1 + e^(-10t + 9.0)))
