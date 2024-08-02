
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>


#include "implementations/matrix_exponential_flop_count.h"

using namespace std;


string flop_count_file = "FLOP_COUNT_FILE_PLACEHOLDER"; // change this to use a different implementation
double *input_matrix;
int input_size;

void read_input_matrix(string file_name, int is_col_major){
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
            int idx = is_col_major? (j*input_size + i) : (i*input_size + j);
            if(! (inFile >> input_matrix[idx])){
                cout << "Error reading from file, terminating program..." << endl;
                exit(1);
            }
        }
    }
    inFile.close();
}


int main(int argc, char **argv) {
    cout << "Starting program." << endl;
    vector<string> matrix_files;
    vector<int> flop_counts;
    int is_col_major = 0;

    // parse command line arguments
    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "--matrix") == 0){
        if(i + 1 >= argc){
            cout << "The flag --matrix expects an argument (path to matrix file)" << endl;
            return -1;
        }
        matrix_files.push_back(argv[i+1]);
        i++;
        }else if(strcmp(argv[i], "--datafolder") == 0){
            if(i + 1 >= argc){
                cout << "The flag --datafolder expects an argument (path to data folder)" << endl;
                return -1;
            }
            string path = argv[i+1];
            for (const auto & entry : filesystem::recursive_directory_iterator(path)){
                if(!entry.is_directory()){
                    matrix_files.push_back(entry.path());
                }
            }
            i++;
        }
        else if(strcmp(argv[i], "--is_col_major") == 0){
            if(i + 1 >= argc){
                cout << "The flag --is_col_major expects 0 or 1" << endl;
                return -1;
            }
            is_col_major = stoi(argv[i+1]);
            i++;
        }
    }

    if(matrix_files.size() < 1){
        cout << "No matrix files provided. Abort." << endl;
        exit(1);
    }

    ofstream file;
    file.open(flop_count_file);
 
    for(const string matrix_file : matrix_files){
        read_input_matrix(matrix_file, is_col_major);
        double *E = (double *) malloc(input_size * input_size * sizeof(double));

        RESET_FLOP_COUNT();
        mat_exp(input_matrix, input_size, E);
        int flop_count = COUNT_FLOPS? get_flop_count() : -1;

        printf("%s - %d flops\n\n", matrix_file.c_str(), flop_count);
        flop_counts.push_back(flop_count);
        file << matrix_file.c_str() << "\t" << flop_count << endl;

        free(E);
        free(input_matrix);
    }

    file.close();
    printf("\n\nOVERVIEW:\n");
    for(int i=0; i < matrix_files.size(); i++){
        printf("%s - %d flops\n", matrix_files.at(i).c_str(), flop_counts.at(i));
    }

    return 0; 
}