
## Setup

Works for Ubuntu 22.04:

1. Install some packages

`> sudo apt-get install libblas-dev liblapack-dev libfmt-dev`   


2. Get Eigen
 - Download files from http://eigen.tuxfamily.org/index.php?title=Main_Page#Download 
 - Extract and copy into a suitable folder (e.g. /usr/local/include/)
 - Make sure PATH_TO_EIGEN in the Makefile is correct


## Benchmarking with Jinja Templates - Workflow

To run the full benchmark generation:
`./run_full_benchmark.sh implementation_name.c unroll_l unroll_c unroll_r dgemm_on loop_order LU_BLAS`
For example:
`./run_full_benchmark.sh exp_test.c 1 1 1 true ikj false` 

This will generate implementations/exp_test.c using the templates as defined in generate.py. 
Additionally, implementations/exp_test_flop_count.txt & implementations/exp_test_benchmark_result.py are generated.



Instead of running the full generation, you can also do it step by step:
1. generate the source file
    1. in jinja/generate.py define the templates that should be used (see variable 'templates')
    2. `> python generate.py arguments`
    3. this should have generated a impl_name.c file in the folder implementations
2. Compute the flop count
    `> ./run_flop_count.sh impl_name.c`
    This generates a impl_name_flop_count.txt file
3. Compile the benchmark files
    1. Make sure debugging and flop couting is disabled!!
    2. `> ./make_benchmark.sh impl_name.c`
    3. The last line prints the compiled impl_name_benchmark.out file
4. Run the benchmarks
    `> ./impl_name_benchmark.out --use_datafolder path_to_matrix_folder`
    Optionally, you can define the following arguments 
    - no_validity: skips the correctness tests
    - min_size, max_size: skips matrices which dimensions are smaller/larger
    - flop_count_file: defines the file which contains the flop counts
    - output_file: defines the output .py file which will containt the benchmark results
5. Generate plots
    - a file called impl_name_benchmark_result.py should have been generated. To generate plots, run
    `> python impl_name_benchmark_result.py`
 

## Benchmarking the base implementation

NOTE: To activate BLAS, set the USEDGEMM flag in the matrix_operations.c file to 1 prior to compiling. To check which cases an execution takes, set the DEBUG flag in matrix_exponential.c to 1 prior to compiling.

To run the benchmarks:

copy-paste main.cpp comp.cpp and flop_count_main.cpp from the folder base_implementation into team01/

### Compute flop count

First, enable the flop count by setting `#define COUNT_FLOPS 1` in implementation_folder/utils.h

`> make clean`

`> make flop_count_main`

`> ./flop_count_main.out --datafolder data/matrixfolder`

Remark: if `#define COUNT_FLOPS 0` the macros are optimized away. That's why we don't have to link against utils.c at all. So these macros should not have an impact on performance when count_flops is disabled.

### Running the benchmark

`> make clean`

`> make benchmark`

`> ./benchmark.out`

There are some command line arguments:
1. --no_validity: disables validity checks
2. --max_size n: sets the maximum input size, expects an integer argument n
3. --min_size n: sets the minimum input size, expects an integer argument n
4. --use_datafolder path_to_data_folder: starts benchmarks with predefined matrices located in .txt files in path_to_data_folder. The max/min size arguments still apply.
5. --output_file filename: defines the filename of the output .txt file (which contains the performance results)


To run the comparisons to the eigen implementation:

`> make clean`

`> make eigen_test`

`> ./eigen_test.out`


## Generating Plots

The benchmarks write all results to a file called your_implementation_benchmark.py. This python file can be used to generate plots.


## Troubleshooting

- Error flop_count(int) is undefined

    - disable flop counting by setting `#define COUNT_FLOPS 0` in implementation_folder/utils.h (or link against utils.c if you need the flop counting)
