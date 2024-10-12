#!/bin/bash
sudo apt-get update
sudo apt-get install -y libfftw3-3 libfftw3-dev libfftw3-doc
git clone https://github.com/QEF/q-e.git
cd q-e
DFLAGS='-D__FFTW3 ' FFT_LIBS='-lfftw3'  ./configure
make pw
# make ph

git clone https://github.com/QMCPACK/qmcpack.git
apt install python3-mpi4py cython3 libxc-dev
cd qmcpack/build
cmake ..
make -j 2
cd qmcpack/external_codes/quantum_espresso
./download_and_patch_qe7.0.sh

cd ../../
./configure --with-hdf5=/content

# Example: https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/QMCPACK_Basic_Example.ipynb
