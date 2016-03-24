#!/bin/bash


#program options
GPU=false
GUI=false
rom=./roms/breakout.bin
epsilon_start=1.0
epsilon_min=0.1
epsilon_decay=0.98
epsilon_explore_idx=10000
replay_memory_capacity=50000
replay_start_size=500
sampleStrategy="uniform"
update_frequency=4
discount_factor=0.98
solver=./prototxt/aleSolver.prototxt
skip_frame=3
show_frame=false
model=null
evaluate=false
eval_epsilon=0.05
target_q_freq=1000



args="--gpu ${GPU} --gui ${GUI} --rom ${rom} --target_q_freq ${target_q_freq} --eval_epsilon ${eval_epsilon} --epsilon_start ${epsilon_start} --epsilon_min ${epsilon_min} --epsilon_decay ${epsilon_decay} --epsilon_explore_idx ${epsilon_explore_idx} --replay_memory_capacity ${replay_memory_capacity} --replay_start_size ${replay_start_size} --sampleStrategy ${sampleStrategy} --update_frequency ${update_frequency} --discount_factor ${discount_factor} --solver ${solver} --skip_frame ${skip_frame} --show_frame ${show_frame} --model ${model} --evaluate ${evaluate}"

echo ${args}

# compile project deepRL 
echo "=============================>Compling project: deepRL"
rm CMakeCache.txt
cp CMakeLists.cpu.txt CMakeLists.txt
cmake .
make clean & make -j2 2> build.errlog

#run , you need set solver_mode: CPU in your solver.prototxt
./deepRL ${args}
