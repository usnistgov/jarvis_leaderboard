#!/bin/bash
apt install python3-mpi4py cython3 libxc-dev gpaw-data
pip -q install gpaw
gpaw test

#Example: https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/GPAW_Colab.ipynb
