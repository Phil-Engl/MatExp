#include <immintrin.h>
#include "benchmark/common.h"
#include "complex.h"

#include "implementations/matrix_exponential.h"

// to include eigen matrix exponential

#include "benchmark/eigen_wrapper.h"
#include "stdio.h"

void baseline_eigen(double *input_matrix, int size, double *output_matrix) {
  eigen_matrix_exp(input_matrix, size, output_matrix); 
}

void own_implementation(double *input_matrix, int size, double *output_matrix) {
  mat_exp(input_matrix, size, output_matrix);
}


/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions() {
  // add_function(&baseline_eigen, "baseline_eigen   ",1);
  add_function(&own_implementation, "IMPLEMENTATION_NAME_PLACEHOLDER",1);
}
