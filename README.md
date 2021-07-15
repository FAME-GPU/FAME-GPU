# FAME-GPU #
The goal of FAME-GPU is to solve the energy band of photonic crystal and get the accurate information of its energy band as soon as possible. For a better computational efficiency, we implemented the FAME-GPU version, it can get more than a hundred times the efficiency of the CPU.  The design of FAME-GPU  allows scientists to download and use it easily, and users can use it without knowing complicated mathematical models and calculation methods.

## Website ##
https://sites.google.com/g2.nctu.edu.tw/fame/

## Installation ##
1. Download **FAME-GPU** form Github.
2. Install **CUDA** and **LAPACK** on your unix system.
3. In Makefile,
> + set the **GPU_TARGET** (Line 15).
> + set the **CUDADIR** (Line 27).
> + set the **LAPACKDIR** (Line 35).
4. Command `make` or `make -j <num>` to compile FAME-GPU. 
5. If successful, you will see the **Main.out** executable file.
6. Before excuting Main.out, user need to set two files: **Popt.txt** and **Material data**.
7. If all settings are completed, command `./Main.out`, FAME-GPU will start working.

## Popt.txt & Material data Setting ##
Please go here https://sites.google.com/g2.nctu.edu.tw/fame/downloads/fame-on-gpu

## Support ##
If you have any questions, please let us know, we are willing to help you.

* Department of Applied Mathematics, National Chiao Tung University, Taiwan
>Jia-Wei Lin : jiawei.am05g@g2.nctu.edu.tw
>Sheng Wang : shengwang.am08g@g2.nctu.edu.tw
* School of Mathematics, Southeast University, Nanjing, China
>Xing-Long Lyu : lxl_math@seu.edu.cn
>Xiao-Ting Zhu-Ge : xtzg@foxmail.com
