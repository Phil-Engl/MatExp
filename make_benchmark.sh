#!/bin/bash

impl_file="$1"
impl_file_without_suffix=${impl_file/\.c/}
flop_count_file=${impl_file/\.c/_flop_count.txt}
benchmark_output_file=${impl_file/\.c/_benchmark_result.py}

echo "INFORMATION:"
echo "Using ${impl_file}"
echo ""
echo "Flop counts are taken from ${flop_count_file} by default"
echo ""
echo "Benchmark result will be stored in ${benchmark_output_file} by default"
echo ""
echo ""

echo "Generating benchmark files..."
python benchmark_generation/replace_implementation.py $1 ${flop_count_file} ${benchmark_output_file}

echo ""
echo "Compiling..."
make benchmark-jinja IMPLEMENTATION_FILE=${impl_file_without_suffix}

echo ""
echo "Runnable: ${impl_file_without_suffix}_benchmark.out"

# run benchmarks using
# > ./${impl_file}_benchmark.out --use_datafolder <path_to_datafolder>