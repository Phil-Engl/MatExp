#!/bin/bash



max_matrix_size=3000

datafolder1=m9
datafolder2=scaled_twice_dense
datafolder3=scaled_twice_lower_triangular

make clean



# Benchmarking base implementation


# echo "\n\n\n\n--------- ${benchmark_name} ----------\n"
rm main.cpp comp.cpp flop_count_main.cpp
# benchmark_name=base_implementation # make sure that this is unique!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# cp base_implementation/main.cpp main.cpp
# cp base_implementation/comp.cpp comp.cpp

# # flop count should be pre-computed in base_implementation/flop_counts.txt already!!!!!!!!!!

# make benchmark IMPLEMENTATION_FOLDER=base_implementation BENCHMARK_OUT=base_implementation/benchmark.out # you can also change compiler and flags

# echo "\n--------- Starting benchmarks for ${benchmark_name} ----------\n"
# taskset 0x1 ./base_implementation/benchmark.out --use_datafolder data/${datafolder1}/ --output_file benchmark_results/${benchmark_name}_${datafolder1}.py --max_size ${max_matrix_size} --is_col_major 0
# taskset 0x1 ./base_implementation/benchmark.out --use_datafolder data/${datafolder2}/ --output_file benchmark_results/${benchmark_name}_${datafolder2}.py --max_size ${max_matrix_size} --is_col_major 0
# taskset 0x1 ./base_implementation/benchmark.out --use_datafolder data/${datafolder3}/ --output_file benchmark_results/${benchmark_name}_${datafolder3}.py --max_size ${max_matrix_size} --is_col_major 0



# Benchmarking a single source file
echo "\n\n\n\n--------- ${benchmark_name} ----------\n"

rm main.cpp comp.cpp flop_count_main.cpp
impl_name=implementations/test_full.c
benchmark_name=test_full # make sure that this is unique!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
impl_file_without_suffix=${impl_name/\.c/}
flop_count_file=${impl_name/\.c/_flop_count}
benchmark_output_file=benchmark_results/${benchmark_name}_results.py
is_col_major=0

echo "\n--------- Computing flop counts ---------\n"
echo "${impl_name} - ${flop_count_file}"
python benchmark_generation/generate_flop_count_impl.py ${impl_name} ${flop_count_file}.txt
make flop_count_single IMPLEMENTATION_FILE=${flop_count_file}
./flop_count_main.out --datafolder data/ --is_col_major ${is_col_major}

echo "\n--------- Generating and compiling benchmark -----------\n"
python benchmark_generation/replace_implementation.py ${impl_name} ${flop_count_file}.txt ${benchmark_output_file} 1
make benchmark-jinja IMPLEMENTATION_FILE=${impl_file_without_suffix} # you can also change compiler and flags

echo "\n--------- Starting benchmarks for ${impl_name} ----------\n"
taskset 0x1 ./${impl_file_without_suffix}_benchmark.out --use_datafolder data/${datafolder1}/ --output_file benchmark_results/${benchmark_name}_${datafolder1}.py --max_size ${max_matrix_size} --is_col_major ${is_col_major}
taskset 0x1 ./${impl_file_without_suffix}_benchmark.out --use_datafolder data/${datafolder2}/ --output_file benchmark_results/${benchmark_name}_${datafolder2}.py --max_size ${max_matrix_size} --is_col_major ${is_col_major}
taskset 0x1 ./${impl_file_without_suffix}_benchmark.out --use_datafolder data/${datafolder3}/ --output_file benchmark_results/${benchmark_name}_${datafolder3}.py --max_size ${max_matrix_size} --is_col_major ${is_col_major}







