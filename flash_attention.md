



Hierarchy of NVIDIA GPU
1. GPU has a bunch of threads to launch kernels
2. Threads are organized into thread blocks
3. Thread blocks are scheduled to run on SMs
4. Inside each blocks, threads are grouped into warps

Excecution Model
1. Threads within a warp can comm. by fast shuffle inst. or cooperate to perform matmul.
2. Warps within a thread block can comm. by R/W to shared memory.

# need to know

1. 基于 roofline model 计算某一个硬件的计算强度时，分子是峰值算力，分母是啥？是 I/O 带宽吗？
2. single excration multi threads?
3. sm/threads/warps 之间的关系