import sys

implementation_file = sys.argv[1] # path to matrix exponenctial implementation
flop_count_file = implementation_file.removesuffix('.c') + '_flop_count.txt'
benchmark_output_file = implementation_file.removesuffix('.c') + '_benchmark_result.py'

if len(sys.argv) > 2:
	flop_count_file = sys.argv[2]
if len(sys.argv) > 3:
	benchmark_output_file = sys.argv[3] 

infile = open("benchmark_generation/comp.cpp", "rt")
outfile = open("comp.cpp", "wt")
for line in infile:
	outfile.write(line.replace('IMPLEMENTATION_NAME_PLACEHOLDER', implementation_file))
infile.close()
outfile.close()

infile = open("benchmark_generation/main.cpp", "rt")
outfile = open("main.cpp", "wt")
for line in infile:
	outfile.write(line.replace('PATH_TO_FLOP_COUNT_FILE_PLACEHOLDER', flop_count_file).replace('PATH_TO_BENCHMARK_OUTPUT_FILE_PLACEHOLDER', benchmark_output_file))
infile.close()
outfile.close()