import sys

implementation_file = sys.argv[1] # path to matrix exponential implementation
flop_count_file = implementation_file.removesuffix('.c') + '_flop_count.txt'

if len(sys.argv) > 2:
	flop_count_file = sys.argv[2]
	

infile = open(implementation_file, "rt")
outfile = open(implementation_file.replace(".c", "_flop_count.c"), "wt")
for line in infile:
	outfile.write(line.replace('#include "matrix_exponential.h"', '#include "matrix_exponential_flop_count.h"'))
infile.close()
outfile.close()
	

infile = open("benchmark_generation/flop_count_main.cpp", "rt")
outfile = open("flop_count_main.cpp", "wt")
for line in infile:
	outfile.write(line.replace("FLOP_COUNT_FILE_PLACEHOLDER", flop_count_file))
infile.close()
outfile.close()
