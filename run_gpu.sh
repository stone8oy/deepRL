#!/bin/bash


#program options
GPU=true
GUI=false
rom=./roms/breakout.bin
epsilon_start=1
epsilon_min=0.1
epsilon_decay=0.99
epsilon_explore_idx=1000
replay_memory_capacity=100000
replay_start_size=10000
sampleStrategy="uniform"
update_frequency=1000
discount_factor=0.98
solver=./prototxt/aleSolver.prototxt
skip_frame=4
show_frame=false
model=./models/rmsprop_nature_iter_100000.caffemodel
evaluate=false



args="--gpu ${GPU} --gui ${GUI} --rom ${rom} --epsilon_start ${epsilon_start} --epsilon_min ${epsilon_min} --epsilon_decay ${epsilon_decay} --epsilon_explore_idx ${epsilon_explore_idx} --replay_memory_capacity ${replay_memory_capacity} --replay_start_size ${replay_start_size} --sampleStrategy ${sampleStrategy} --update_frequency ${update_frequency} --discount_factor ${discount_factor} --solver ${solver} --skip_frame ${skip_frame} --show_frame ${show_frame} --model ${model} --evaluate ${evaluate}"

echo ${args}

# compile project deepRL 
echo "=============================>Compling project: deepRL"
rm CMakeCache.txt
cp CMakeLists.gpu.txt CMakeLists.txt
cmake .
make clean & make -j2 2> build.errlog

#run , you need set solver_mode: GPU in your solver.prototxt
./deepRL ${args}
