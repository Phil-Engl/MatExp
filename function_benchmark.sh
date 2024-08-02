#!/bin/bash

#sed -i "/#define USEDGEMM/c\#define USEDGEMM 1" base_implementation/matrix_operations.c 
#make benchmark-function-base
#taskset 0x1 ./benchmark_functions.out 0 6 1 1 1 1 1 1 1 | tee bm_implementations/results/base_implementation_blas.csv

sed -i "/#define USEDGEMM/c\#define USEDGEMM 0" base_implementation/matrix_operations.c 
make benchmark-function-base
taskset 0x1 ./benchmark_functions.out 5 6 1 0 0 1 1 1 1 | tee bm_implementations/results/base_implementation_noblas.csv

#cd jinja
#python3 generate_benchmark.py 1 1 1 false false
#cd ..
#make benchmark-optimized
#taskset 0x1 ./optimize_version_benchmark.out 0 6 1 1 1 1 1 1 1 | tee bm_implementations/results/runtime_1_1_1_noblas.csv

#cd jinja
#python3 generate_benchmark.py 4 4 1 false false
#cd ..
#make benchmark-optimized
#taskset 0x1 ./optimize_version_benchmark.out 0 6 0 1 1 0 1 1 0 | tee bm_implementations/results/runtime_4_4_1_noblas.csv

#cd jinja
#python3 generate_benchmark.py 8 8 1 false false
#cd ..
#make benchmark-optimized
#taskset 0x1 ./optimize_version_benchmark.out 0 7 0 1 1 0 1 1 0 | tee bm_implementations/results/runtime_8_8_1_noblas.csv

#cd jinja
#python3 generate_benchmark.py 1 1 1 true false
#cd ..
#make benchmark-optimized
#taskset 0x1 ./optimize_version_benchmark.out 0 6 1 0 0 1 1 1 1 | tee bm_implementations/results/runtime_1_1_1_blas.csv

#cd jinja
#python3 generate_benchmark.py 4 4 1 true false
#cd ..
#make benchmark-optimized
#taskset 0x1 ./optimize_version_benchmark.out 0 6 0 0 0 0 1 1 0 | tee bm_implementations/results/runtime_4_4_1_blas.csv

#cd jinja
#python3 generate_benchmark.py 8 8 1 true false
#cd ..
#make benchmark-optimized
#taskset 0x1 ./optimize_version_benchmark.out 0 7 0 0 0 0 1 1 0 | tee bm_implementations/results/runtime_8_8_1_blas.csv

#cd bm_implementations/results
#python3 plot.py

