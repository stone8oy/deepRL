# deepRL
deep-q-learning for atari games with caffe 

Referred impls
-------------
https://github.com/spragunr/deep_q_rl

https://github.com/muupan/dqn-in-the-caffe

Install dependencies
--------------------
caffe required : https://github.com/BVLC/caffe

bash setup.sh （only install the ALE environment）

Build for CPU
-------
bash build.sh 

Build for GPU
-------
you should turn gpu-related options on in CMakeLists.txt, then run : bash build.sh

Run
---
[cpu] bash run_cpu.sh 

[gpu]run for gpu is similar, you need build a gpu version first!

cp run_cpu.sh run_gpu.sh  

set GPU=true in run_gpu.sh 

bash run_gpu.sh
