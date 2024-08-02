#pragma once

#include <string>

typedef void(*comp_func)(double *input_matrix, int size, double *output_matrix);
void add_function(comp_func f, std::string name, int flop);
