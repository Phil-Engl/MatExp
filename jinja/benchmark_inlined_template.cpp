
//benchmark_template

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <filesystem>
#include <fmt/core.h>
#include <immintrin.h>
#include <string>
#include <vector>



#include "../benchmark/tsc_x86.h"
#include "../benchmark/utils.h"
#define CYCLES_REQUIRED 1e8
#define REP 50



double mmm_benchmark(const double* A, const double *B, double *C, int m, int n, int t){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;
   
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (long i = 0; i < num_runs; i++) {
            mmm(A, B, C, m, n, t);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            mmm(A, B, C, m, n, t);        
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;
  
    return  cycles;
}

// Base operations function benchmark

// MM add
double scalar_matrix_mult_benchmark(double alpha, const double *A, double *C, int m, int n){
     double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;
   
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (long i = 0; i < num_runs; i++) {
            scalar_matrix_mult(alpha, A, C, m, n);    
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            scalar_matrix_mult(alpha, A, C, m, n);           
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;

    return  cycles;
}

// One Norm

double onenorm_benchmark(const double* A, int m, int n){
     double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;

    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            max_val = onenorm(A,  m, n);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            max = onenorm( A, m, n);          
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;

    return  cycles;

}

// normest
double onenormest_benchmark(const double *A, int n){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;
 
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            max_val = normest(A,  n);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            max = normest(A,   n);       
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return  cycles; 

}

// Eval 3.4

double eval3_4_m3_benchmark(const double* A, const double* A_2, int n, double *P_3, double *Q_3){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m3(A, A_2, n, P_3, Q_3);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m3(A, A_2, n, P_3, Q_3);   
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
   
    return  cycles;     
}


double eval3_4_m5_benchmark(const double* A, const double* A_2, const double* A_4, int n, double *P_5, double *Q_5){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m5(A, A_2, A_4, n, P_5, Q_5);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m5(A, A_2, A_4, n, P_5, Q_5);   
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return  cycles;     
}

double eval3_4_m7_benchmark(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, double *P_7, double *Q_7){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m7(A, A_2, A_4, A_6, n, P_7, Q_7);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m7(A, A_2, A_4, A_6, n, P_7, Q_7);  
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return  cycles;     
}

double eval3_4_m9_benchmark(const double* A, const double* A_2, const double* A_4, const double* A_6, const double* A_8, int n, double *P_9, double *Q_9){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m9(A, A_2, A_4, A_6, A_8, n, P_9, Q_9);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m9(A, A_2, A_4, A_6, A_8, n, P_9, Q_9);  
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
 
    return  cycles;     
}



// Eval 3_5

double eval3_5_benchmark(const double* A, double *A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_5(A,  A_2, A_4, A_6, n, P_13, Q_13);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_5(A,  A_2, A_4, A_6, n, P_13, Q_13);      
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return cycles;     
}

// Eval 3_6
double eval3_6_benchmark(double * P_m, double *Q_m, int n, double *R_m, int triangular_indicator){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_6(P_m,  Q_m, n, R_m , triangular_indicator);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_6(P_m,  Q_m, n, R_m , triangular_indicator);  
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
   
    return  cycles;  
}

int readmatrix(double *A, int n, const char* path){
    FILE *fptr = fopen(path, "r");
    if(fptr==NULL){
        printf("file not found\n");
        return 0;
    }
    char buf[50];
    fscanf(fptr, "%s", buf);
    int m = atoi(buf);
    fscanf(fptr, "%s", buf);
    int k = atoi(buf);
    if(m!=k){
        printf("non square matrix!\n");
        return 0;
    }
    if(m != n){
        printf("wrong size\n");
        return 0;
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j<n; j++){
            fscanf(fptr, "%s", buf);
            A[j * n + i] = atof(buf);
        }
    }
    return 1;
}



int main (int argc, char* argv[]) {
    printf("Function,#cycles,n,floats/cycle\n");

    int lower = 0;
    int upper = 4;
    int numpaths = 7;

    if(argc < 5){
        printf("mmm mm_add onenorm eval3_4 eval3_5\n");
        return 0;
    }
    int mmm_on = atoi(argv[1]);
    int scm_on = atoi(argv[2]);
    int onenorm_on = atoi(argv[3]);
    int eval3_4_on = atoi(argv[4]);
    int eval3_5_on = atoi(argv[5]);
    
    int sizes[] = {32,64,128,256,512,768,1024};
    const char **paths = (const char**)malloc(10 * sizeof(const char*));
    paths[0] = "data/scaled_twice_dense/dense_0032.txt";
    paths[1] = "data/scaled_twice_dense/dense_0064.txt";
    paths[2] = "data/scaled_twice_dense/dense_0128.txt";
    paths[3] = "data/scaled_twice_dense/dense_0256.txt";
    paths[4] = "data/scaled_twice_dense/dense_0512.txt";
    paths[5] = "data/scaled_twice_dense/dense_0768.txt";
    paths[6] = "data/scaled_twice_dense/dense_1024.txt";

    double** mats = (double**)malloc(10* sizeof(double*));

    
    for(int i = 0; i < numpaths; i++){
        mats[i] = (double *) aligned_alloc(32, sizes[i]*sizes[i]*sizeof(double));
        if(!readmatrix(mats[i],sizes[i],paths[i])){
            return 0;
        }
        
    }
    
    for(int i = lower; i < upper; i++){
        long n = sizes[i];
        //mmm benchmark
        double *A = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *B = (double *) aligned_alloc(32, n*n*sizeof(double));
        double* C = (double *) aligned_alloc(32, n*n*sizeof(double));

        double *A_2 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *A_4 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *A_6 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *A_8 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *P_m = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *Q_m = (double *) aligned_alloc(32, n*n*sizeof(double));
        if(mmm_on || scm_on){
            memcpy(A, mats[i], n*n*sizeof(double));
            memcpy(B, mats[i], n*n*sizeof(double));
        }

        double res = 0;
        double flops = 0;
        if(mmm_on){   
            res = mmm_benchmark(A,B,C,n,n,n);
            flops = 2*n*n*n;
            printf("mmm, %.4f, %4d, %.4f\n", res, n, flops/res);
        }

       
        //mm_add benchmark
        if(scm_on){
            res = scalar_matrix_mult_benchmark(2.5,A,C,n,n);
            flops = n*n;
            printf("scalar_mm, %.4f, %4d, %.4f\n", res, n, flops/res);  
        }
       
        //onenorm benchmark
        if(onenorm_on){
            res = onenorm_benchmark(A,n,n);
            flops = 2*n*n;
            printf("onenorm, %.4f, %4d, %.4f\n", res, n, flops/res);
        }

        if(eval3_4_on || eval3_5_on){
            memcpy(A, mats[i], n*n*sizeof(double));
            memcpy(A_2, mats[i], n*n*sizeof(double));
            memcpy(A_4, mats[i], n*n*sizeof(double));
            memcpy(A_6, mats[i], n*n*sizeof(double));
            memcpy(A_8, mats[i], n*n*sizeof(double));
        }        


        if(eval3_4_on){
            //flops = 6*n*n + 2*n*n*n;
            //res = eval3_4_m3_benchmark(A, A_2, n, P_m, Q_m);
            //printf("eval3_4_m3, %.4f, %4d, %.4f\n", res, n, flops/res);
//
            //flops = 10*n*n + 2*n*n*n;
            //res = eval3_4_m5_benchmark(A, A_2, A_4, n, P_m, Q_m);
            //printf("eval3_4_m5, %.4f, %4d, %.4f\n", res, n, flops/res);
            //
            //flops = 14*n*n + 2*n*n*n;
            //res = eval3_4_m7_benchmark(A, A_2, A_4, A_6, n, P_m, Q_m);
            //printf("eval3_4_m7, %.4f, %4d, %.4f\n", res, n, flops/res);        
           
            flops = 18*n*n + 2*n*n*n;
            res = eval3_4_m9_benchmark(A, A_2, A_4, A_6, A_8, n, P_m, Q_m);
            printf("eval3_4, %.4f, %4d, %.4f\n", res, n, flops/res);
            
        }

        if(eval3_5_on){
            flops = 26*n*n + 6*n*n*n;
            res = eval3_5_benchmark(A, A_2, A_4, A_6, n, P_m, Q_m);
            printf("eval3_5, %.4f, %4d, %.4f\n", res, n, flops/res);        
        }

        free(A);
        free(B);
        free(C);
        free(A_2);
        free(A_4);
        free(A_6);
        free(A_8);
        free(P_m);
        free(Q_m);
    }

    return 1;

}