#!/bin/bash

impl_name=$1
impl_file_without_suffix=${impl_name/\.c/}
unroll_l=$2 #linear unrollings (n*n)
unroll_c=$3 #column wise unrollings (eg at 4x4 matrix, use 4 for full unrolling)
unroll_r=$4 #row wise unrolling (e.g at 4x4 matrix use 1 for full unrolling)
dgemm_on=$5 # true or false
debug=$6 # 0 or 1
loop_order=$7 #either ikj or kij
LU_BLAS=$8 # true or false
fc=$9 # true to include FLOP_COUNT_INC macros
profiling=$10
is_col_major=0

make clean

rm main.cpp comp.cpp flop_count_main.cpp

cd jinja/

python generate.py ${unroll_l} ${unroll_c} ${unroll_r} ${dgemm_on} ${debug} ${loop_order} ${LU_BLAS} ${fc} ${profiling} ${impl_name}

cd ..

./run_flop_count.sh implementations/${impl_name} ${is_col_major}

./make_benchmark.sh implementations/${impl_name}

./implementations/${impl_file_without_suffix}_benchmark.out --use_datafolder data/m5/ --is_col_major ${is_col_major} --output_file benchmark_results/${impl_file_without_suffix}.py