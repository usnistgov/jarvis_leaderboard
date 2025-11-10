#!/bin/bash

#get julia
if [ ! -f julia-1.10.2/bin/julia ]
then
    echo "get julia"
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.2-linux-x86_64.tar.gz
    gunzip julia-1.10.2-linux-x86_64.tar.gz
    tar -xvf julia-1.10.2-linux-x86_64.tar
fi

#run test
./julia-1.10.2/bin/julia run_latt.jl

echo "done run.sh"

