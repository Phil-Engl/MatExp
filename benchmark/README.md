`> g++ -I /usr/local/include/eigen-3.4.0/ -o benchmark_out benchmark/eigen_wrapper.cpp main.cpp comp.cpp base_implementation/matrix_operations.c base_implementation/onenorm.c base_implementation/matrix_exponential.c -lm -lcblas -lblas -Wall`


To include eigen : git clone https://gitlab.com/libeigen/eigen.git

and make sure to have the correct path when compiling (see Makefile)