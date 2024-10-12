#!/bin/bash
sudo apt-get update
sudo apt-get install libxc-dev
sudo apt install python3-mpi4py cython3 libxc-dev gpaw-data
conda install gpaw -c conda-forge
#pip -q install ase 
#pip -q install gpaw
gpaw test

#Example: https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/GPAW_Colab.ipynb
