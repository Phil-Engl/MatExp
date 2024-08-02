#!/bin/bash

impl_file="$1"
flop_count_file="${impl_file/\.c/_flop_count}"
is_col_major=$2

python benchmark_generation/generate_flop_count_impl.py $1 ${flop_count_file}.txt

make flop_count_single IMPLEMENTATION_FILE=${flop_count_file}

./flop_count_main.out --datafolder data/ --is_col_major ${is_col_major}