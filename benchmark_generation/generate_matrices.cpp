#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fmt/core.h>
#include <filesystem>

#include "../benchmark/utils.h"

using namespace std;


void write_results_to_file(double* A, int n, string filename){
  ofstream file;
  file.open(filename);
  file << n << " " << n << endl;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
        file << A[i*n+j] << " ";
    }
    file << endl;
  }
  file.close();
}

int main(int argc, char **argv) {
    std::string path = "../data/";

    int divisor = 100;
    // parse command line arguments
    for(int i = 1; i < argc; i++){
        divisor = stoi(argv[i]);
    }
    double *A; 
    double value ;
    int sizes[] = {16, 32, 48, 64, 96, 128, 256, 256+128, 512, 512+256, 1024};

    for(int i = 0; i < sizeof(sizes) / sizeof(int); i++){
        int n = sizes[i];
        A = (double *)malloc(n*n*sizeof(double));
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                value = ((rand() % 40)-15) / (1.0*divisor);
                A[i*n+j] = value;
            }
        }
        string filename = fmt::format("../matrices/dense_{}.txt", n);
        if(n < 10){
            filename = fmt::format("../matrices/dense_000{}.txt", n);
        }else if(n < 100){
            filename = fmt::format("../matrices/dense_00{}.txt", n);
        }else if(n < 1000){
            filename = fmt::format("../matrices/dense_0{}.txt", n);
        }
        write_results_to_file(A, n, filename);
        free(A);
    }

    return 0;
}