/**
*      _________   _____________________  ____  ______
*     / ____/   | / ___/_  __/ ____/ __ \/ __ \/ ____/
*    / /_  / /| | \__ \ / / / /   / / / / / / / __/
*   / __/ / ___ |___/ // / / /___/ /_/ / /_/ / /___
*  /_/   /_/  |_/____//_/  \____/\____/_____/_____/
*
*  http://www.acl.inf.ethz.ch/teaching/fastcode
*  How to Write Fast Numerical Code 263-2300 - ETH Zurich
*  Copyright (C) 2019 
*                   Tyler Smith        (smitht@inf.ethz.ch) 
*                   Alen Stojanov      (astojanov@inf.ethz.ch)
*                   Gagandeep Singh    (gsingh@inf.ethz.ch)
*                   Markus Pueschel    (pueschel@inf.ethz.ch)
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program. If not, see http://www.gnu.org/licenses/.
*/

#include <list>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <filesystem>
#include <fmt/core.h>
#include "benchmark/tsc_x86.h"
#include "benchmark/utils.h"
#include "benchmark/eigen_wrapper.h"

using namespace std;

#define NR 32
#define CYCLES_REQUIRED 1e8
#define REP 1
#define EPS (1e-3)
#define TEST_SIZE (8)

string flop_count_file = "base_implementation/flop_counts.txt";
string benchmark_filename = "base_implementation/benchmark_results.py";

void kernel_base(double *input_matrix, int size, double *output_matrix) {
    eigen_matrix_exp(input_matrix, size, output_matrix);
    }


/* prototype of the function you need to optimize */
typedef void(*comp_func)(double* input_matrix, int size, double *ouput_matrix);

void   register_functions();
double get_perf_score(comp_func f);
double perf_test_single(comp_func f, string desc, int flops, double* inp_mat, int s);
void   perf_test();
void   add_function(comp_func f, string name, int flop);


int MIN_SIZE = 2;
int MAX_SIZE = 256;

int use_datafolder = 0;
vector<int> sizes;

/* Global vars, used to keep track of student functions */
vector<comp_func> userFuncs;
vector<string> funcNames;
vector<int> funcFlops;
int numFuncs = 0;

vector<string> datafolder_files;

double* input_matrix;
int input_size;

/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(comp_func f, string name, int flops) {
    userFuncs.push_back(f);
    funcNames.emplace_back(name);
    funcFlops.push_back(flops);
    numFuncs++;
}

int get_num_test_matrices(){
  if(use_datafolder){
    return datafolder_files.size();
  }else{
    return sizes.size();
  }
}

void read_input_matrix(string file_name){
    ifstream inFile;
    inFile.open(file_name);
    int cols;

    if(! (inFile >> input_size)){
      cout << "Error reading from file, terminating program..." << endl;
      exit(1);
    }
    if(! (inFile >> cols)){
      cout << "Error reading from file, terminating program..." << endl;
      exit(1);
    }
    if(input_size != cols){
      cout << "Error reading from file, input is not a square matrix..." << endl;
      exit(1);
    }

    input_matrix = (double*) aligned_alloc(32, input_size*input_size*sizeof(double));
    
    for(int i = 0; i < input_size; i++) {
        for(int j = 0; j < input_size; j++) {
            if(! (inFile >> input_matrix[i*input_size + j])){
                cout << "Error reading from file, terminating program..." << endl;
                exit(1);
            }
        }
    }
    inFile.close();
}

void generate_input_matrix(int n){
    input_matrix = (double *)aligned_alloc(32, n*n*sizeof(double));
    rands(input_matrix, n, n);
    input_size = n;
}

void fill_input_matrix(size_t idx){
  if(use_datafolder && idx < datafolder_files.size()){
    read_input_matrix(datafolder_files[idx]);
  }else if(idx < sizes.size()){
    generate_input_matrix(sizes[idx]);
  }else{
    printf("Could not generate input matrix for idx %ld\n", idx);
    exit(1);
  }
}

string get_matrix_name(size_t idx){
  if(use_datafolder && idx < datafolder_files.size()){
    return datafolder_files[idx];
  }else if(idx < sizes.size()){
    return fmt::format("random n={}", sizes[idx]);
  }else{
    return "invalid matrix";
  }
}

//Check validity of functions
void check_validity(){
  size_t num_tests = get_num_test_matrices();
  for(size_t mat = 0; mat < num_tests; mat++){

    fill_input_matrix(mat);
    if(input_size < MIN_SIZE || input_size > MAX_SIZE){
      continue;
    }
    printf("Testing with matrix %s...\n", get_matrix_name(mat).c_str());

    double *out_mat_base = (double *)aligned_alloc(32, input_size*input_size*sizeof(double));
    double *out_mat1 = (double *)aligned_alloc(32, input_size*input_size*sizeof(double));
    
    kernel_base(input_matrix, input_size, out_mat_base);
    /*
    printf("Base matrix : \n");
    for(int i = 0 ; i < 5 ; i++){
      for(int j = 0 ; j < 5; j++){
        printf(" %f | ", out_mat_base[i*5+j]);
      }
      printf("---------- \n");
    }
    */

    for (int i = 0; i < numFuncs; i++) {
      comp_func f = userFuncs[i];
      f(input_matrix, input_size, out_mat1);
      /*
      printf("not base matrix : \n");
    for(int i = 0 ; i < 5 ; i++){
      for(int j = 0 ; j < 5; j++){
        printf(" %f |", out_mat1[i*5+j]);
      }
      
      printf("---------- \n");
    }
    */
      
      double error = nrm_sqr_diff(out_mat1, out_mat_base, input_size*input_size);
      printf("error = %f \n", error);
      if (error > EPS) {
        cout << "\033[1;31m" << "The result of the " << i+1 << "th function is not correct." << "\033[0m" << std::endl;
      }
    }
    printf("\n");

    // Free memory
    free(input_matrix);
    free(out_mat1);
    free(out_mat_base);
  }
}


void read_flop_counts(vector<string> *matrices_flop_count, vector<long>* flop_counts){
  ifstream inFile;
  inFile.open(flop_count_file);
  string matrix_name;
  int flop_count;
  while((inFile >> matrix_name)){
    if(!(inFile >> flop_count)){
      cout << "Error reading from file, terminating program..." << endl;
      inFile.close();
      exit(1);
    }
    matrices_flop_count->push_back(matrix_name);
    flop_counts->push_back(flop_count);
  }
  inFile.close();
}

long get_flop_count(string matrix_name, vector<string> matrices_flop_count, vector<long> flop_counts){
  for(int i=0; i<matrices_flop_count.size(); i++){
    if(!strcmp(matrices_flop_count.at(i).c_str(), matrix_name.c_str())){
      return flop_counts.at(i);
    }
  }
  return -1;
}

// writes the performance results into a file using a format that can be parsed easily
void write_results_to_file(double* runtime_results, double* perf_results, long* flop_counts){
  ofstream file;
  file.open(benchmark_filename);
  file << "import plot\n" << endl;
  file << "# Performance results\n" << endl;
  int num_tests = get_num_test_matrices();

  file << "columns = [";
  for (int i = 0; i < num_tests; i++) {
    file << "\"" << get_matrix_name(i) << "\"" << ", ";
  }
  file << "] # input matrices" << endl;

  file << "rows = [";
  for (int i = 0; i < numFuncs; i++) {
    file << "\"" << funcNames[i] << "\", ";
  }
  file << "] # function names\n" << endl;


  file << "runtime_results = [" << endl;
  for (int i = 0; i < numFuncs; i++) {
    file << "[";
    for(int j = 0; j < num_tests; j++){
      file << runtime_results[i * num_tests + j] << ", ";
    }
    file << "], " << endl;
  }
  file << "]" << endl;

  file << "perf_results = [" << endl;
  for (int i = 0; i < numFuncs; i++) {
    file << "[";
    for(int j = 0; j < num_tests; j++){
      file << perf_results[i * num_tests + j] << ", ";
    }
    file << "], " << endl;
  }
  file << "]" << endl;

  file << "flops = [" << endl;
  for (int i = 0; i < numFuncs; i++) {
    file << "[";
    for(int j = 0; j < num_tests; j++){
      file << flop_counts[i * num_tests + j] << ", ";
    }
    file << "], " << endl;
  }
  file << "]\n" << endl;
  file << "plot.make_all_plots(columns, rows, runtime_results, perf_results, flops)\n" << endl;

  file.close();
}


/*
* runs the performance benchmark for a single function and input matrix
*/
double perf_test_single(comp_func f, string desc, int flops){
    double cycles = 0.;
    long num_runs = 30;
    double multiplier = 1;
    myInt64 start, end;
    double* out_mat = (double *)aligned_alloc(32, input_size*input_size*sizeof(double));

    // Warm-up phase: we determine a number of executions that allows
    // the code to be executed for at least CYCLES_REQUIRED cycles.
    // This helps excluding timing overhead when measuring small runtimes.
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (long i = 0; i < num_runs; i++) {
            f(input_matrix, input_size, out_mat);           
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);

    list<double> cyclesList;

    // Actual performance measurements repeated REP times.
    // We simply store all results and compute medians during post-processing.
    double total_cycles = 0;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            f(input_matrix, input_size, out_mat);           
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

        cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    free(out_mat);
    return  cycles;
}

/*
* runs performance benchmarks
*/
void perf_test() {
   // print header
  cout << "\nBenchmarking..." << endl;
  cout << "Input matrix\t";
  for (int i = 0; i < numFuncs; i++) {
      cout << funcNames[i] << "\t";
  }
  cout << endl;

  int num_tests = get_num_test_matrices();
  double* runtime_results = (double *) aligned_alloc(32, numFuncs*num_tests*sizeof(double));
  double* perf_results = (double *) aligned_alloc(32, numFuncs*num_tests*sizeof(double));
  long* flops_output = (long *) aligned_alloc(32, numFuncs*num_tests*sizeof(long));
  vector<string> matrices_flop_count;
  vector<long> flop_counts;
  read_flop_counts(&matrices_flop_count, &flop_counts);

  for(int s = 0; s < num_tests; s++){
    cout << get_matrix_name(s) << "\t";
    long flops = get_flop_count(get_matrix_name(s), matrices_flop_count, flop_counts);
  
    // same input matrix for all functions such that runtime is comparable!
    fill_input_matrix(s);
    if(input_size < MIN_SIZE || input_size > MAX_SIZE){
      cout << " invalid size \t\t" << endl;
      for (int i = 0; i < numFuncs; i++) {
        runtime_results[i*num_tests + s] = -1;
        perf_results[i*num_tests + s] = -1;
        flops_output[i*num_tests + s] = -1;
      }
      continue;
    }

    for (int i = 0; i < numFuncs; i++) {
      double runtime = perf_test_single(userFuncs[i], funcNames[i], 1);
      cout << runtime << " cycles, " << flops/runtime << " flops/cycle\t";
      runtime_results[i*num_tests + s] = runtime;
      perf_results[i*num_tests + s] = flops/runtime;
      flops_output[i*num_tests + s] = flops;
    }
    cout << endl;
    free(input_matrix);
  }
  
  write_results_to_file(runtime_results, perf_results, flops_output);
  free(runtime_results);
  free(perf_results);
  free(flops_output);
}

int main(int argc, char **argv) {
  cout << "Starting program. ";
  int execute_validity_check = 1;

  // parse command line arguments
  for(int i = 1; i < argc; i++){
    if(strcmp(argv[i], "--no_validity") == 0){
      execute_validity_check = 0;
    }if(strcmp(argv[i], "--use_datafolder") == 0){
      use_datafolder = 1;
      if(i + 1 >= argc){
        cout << "The flag --use_datafolder expects an argument (path to data folder)" << endl;
        return -1;
      }
      string path = argv[i+1];
      for (const auto & entry : filesystem::directory_iterator(path)){
        datafolder_files.push_back(entry.path());
      }
      sort(datafolder_files.begin(), datafolder_files.end());
      i++;
    }else if(strcmp(argv[i], "--max_size") == 0){
      if(i + 1 >= argc){
        cout << "The flag --max-size expects an integer argument" << endl;
        return -1;
      }
      MAX_SIZE = stoi(argv[i+1]);
      i++;
    }
    else if(strcmp(argv[i], "--min_size") == 0){
      if(i + 1 >= argc){
        cout << "The flag --min-size expects an integer argument" << endl;
        return -1;
      }
      MIN_SIZE = stoi(argv[i+1]);
      i++;
    }
    else if(strcmp(argv[i], "--output_file") == 0){
      if(i + 1 >= argc){
        cout << "The flag --output_file expects a string argument" << endl;
        return -1;
      }
      benchmark_filename = argv[i+1];
      i++;
    }
    else if(strcmp(argv[i], "--flop_count_file") == 0){
      if(i + 1 >= argc){
        cout << "The flag --flop_count_file expects a string argument" << endl;
        return -1;
      }
      flop_count_file = argv[i+1];
      i++;
    }
  }

  for(int i = MIN_SIZE; i <= MAX_SIZE; i=i*2){
    sizes.push_back(i);
  }

  register_functions();

  if (numFuncs == 0){
    cout << endl;
    cout << "No functions registered - nothing for driver to do" << endl;
    cout << "Register functions by calling register_func(f, name)" << endl;
    cout << "in register_funcs()" << endl;

    return 0;
  }
  cout << numFuncs << " functions registered." << endl;
  
  if(execute_validity_check){
    cout << "Checking validity..." << endl;
    check_validity();
  }else{
    cout << "\033[1;31mSkipping validity checks!\033[0m" << endl;
  }

  perf_test();
  
  return 0;
}