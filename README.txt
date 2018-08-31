necessary directories
bin,obj,src

Cuda/c++/c dependencies
libtiff,cuda-9
-get Python.h flags for linking and compiling by doing This
  $ /path/to/bin/python3.n-config --cflags
  $ /path/to/bin/python3.n-config --ldflags

python dependencies
python-3.5,tensorflow-1.10,numpy,scikit-images,pandas


NOTE change location of libraries in Makefile
