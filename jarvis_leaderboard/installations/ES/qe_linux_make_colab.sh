#!/bin/bash
sudo apt-get update
sudo apt-get install -y libfftw3-3 libfftw3-dev libfftw3-doc
git clone https://github.com/QEF/q-e.git
cd q-e
DFLAGS='-D__FFTW3 ' FFT_LIBS='-lfftw3'  ./configure
make pw
# make ph
# Example: https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/JARVIS_QuantumEspressoColab_Basic_Example.ipynb
