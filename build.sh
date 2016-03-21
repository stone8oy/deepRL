#!/bin/bash

# compile project deepRL 
echo "=============================>Compling project: deepRL"
rm CMakeCache.txt
cmake .
make clean & make -j2 2> build.errlog

