/* --- testing template ---*/

#include "benchmark/eigen_wrapper.h"




#define EPS (1e-3)
/**
 * @brief generates a random integer in [lower,upper]
 * 
 * @param lower lower bound
 * @param upper upper bound
 * @return int random number
 */
int rnd(int lower, int upper){
    return (rand() % (upper - lower + 1)) + lower; 
}

/**
 * @brief fills a matrix with random numbers or a constant with a given sparsity
 * 
 * @param A The matrix to be filled
 * @param n the dimensions of the input matrix
 * @param sparsity reciprocal faction of numbers to fill (e.g. 5 => 1/5 of the numbers will not be zero)
 * @param random 1 if the numbers should be in the given range, 0 if lower should be treated as the constant to be filled in
 * @param lower lower bound of random fill or const
 * @param upper upper bound of random fill
 */
void fill_matrix(double *A, int n, int sparsity, int random, double lower, double upper){
    if(sparsity <= 0){
        printf("Choose sparsity >= 1");
        return;
    }else if (sparsity == 1){
        if(random){
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    A[i * n + j] = (double)rnd((int)lower,(int)upper);
                    if(A[i * n + j] == 0.0){
                        j--;
                    }
                }
            }
        }else{
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    A[i * n + j] = lower;
                }
            }
        }
    }else{
         int rand = 0;
         if(random){
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    rand = rnd(0,sparsity - 1);
                    if(rand){
                        A[i * n + j] = 0.0;
                    }else{
                        A[i * n + j] = (double)rnd((int)lower,(int)upper);
                    }
                }
            }
        }else{
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    rand = rnd(0,sparsity - 1);
                    if(rand){
                        A[i * n + j] = 0.0;
                    }else{
                        A[i * n + j] = lower;
                    }
                }
            }
        }
    }
    return;
}

void fill_fract(double *A, int n, int sparsity, int max, int prec){
    if(sparsity == 0){
        printf("Choose sparsity >= 1");
        return;
    }else if(sparsity == 1){
        for(int i = 0; i < n; i++){
            for(int j = 0; j <n; j++){
                A[i * n + j] = (double)rnd(-max, max) / (double)prec;
            }
        }
    }else{
        int rand = 0;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                rand = rnd(0,sparsity - 1);
                if(rand){
                    A[i * n + j] = 0.0;
                }else{
                    A[i * n + j] = (double)rnd(-max, max) / (double)prec;
                }
            }
        }
    }

}

/**
 * @brief Turns a previously filled matrix into a lower triangular matrix
 * 
 * @param A The matrix to be transformed
 * @param n The dimensions of the matrix
 */
void make_lower_triangular(double *A, int n){
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            A[i * n + j] = 0.0;
        }
    }
    return;
}


/**
 * @brief Turns a previously filled matrix into an upper triangular matrix
 * 
 * @param A The matrix to be transformed
 * @param n The dimensions of the matrix
 */
void make_upper_triangular(double *A, int n){
    for(int i = 1; i < n; i++){
        for(int j = 0; j < i; j++){
            A[i * n + j] = 0.0;
        }
    }
    return;
}

void transpose_in_place(double* A, int n){
    double tmp = 0;
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            tmp = A[i * n + j];
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = tmp;

        }
    }
}

void transpose(double *A, double *B, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            B[i * n + j] = A[j * n + i];
        }
    }
}

void printmatrix(const double* M, int m, int n){
    // FLOP_COUNT_INC(0, "printmatrix");
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("%f, ", M[i*n + j]);
        }
        printf("\n");
    }
}

double nrm_sqr_diff(double *x, double *y, int n) {
    double nrm_sqr = 0.0;
    for(int i = 0; i < n; i++) {
        nrm_sqr += (x[i] - y[i]) * (x[i] - y[i]);
    }
    
    return nrm_sqr;
}

/**
 * @brief Compares the results of the two implementations
 * 
 * @param A input matrix A
 * @param B input matrix B
 * @param n dimensions of both matrices
 * @return int 1 if they are the same with a given precission, 0 otherwise
 */
int compare_results(double *A, double *B, int n){
    double error = nrm_sqr_diff(A, B, n*n);
    if(error > EPS){
        return 0;
    }
    return 1;
}

int main(){
    srand(time(0));
    for(int n = 8; n <= 8; n+=4){
        double* A = (double*)aligned_alloc(32, n*n*sizeof(double));
        double* A_c = (double*)aligned_alloc(32, n*n*sizeof(double));
        double* B = (double*)aligned_alloc(32, n*n*sizeof(double));
        double* C = (double*)aligned_alloc(32, n*n*sizeof(double));
        //fill_matrix(A,n,1,1,-2.0,2.0);
        fill_fract(A,n,1,100,1000);
        //printmatrix(A,n,n);
        make_lower_triangular(A,n);

        eigen_matrix_exp(A,n,B);
        
        transpose(A, A_c, n);
        mat_exp(A_c,n,C);
        transpose_in_place(C,n);
        //printmatrix(B,n,n);
        printf("\n");
        //printmatrix(C,n,n);
        if(compare_results(B,C,n)){
            printf("Random dense matrix -2 - 2 success! n = %d\n", n);
        }else{
            printf("Random dense matrix -2 - 2 fail! n = %d\n", n);
            printf("A:\n");
            printmatrix(A,n,n);
            printf("Eigen result:\n");
            printmatrix(B,n,n);
            printf("Mat exp result:\n");
            printmatrix(C,n,n);
            free(A);
            free(B);
            free(C);
            return 1;
        }
        free(A);
        free(A_c);
        free(B);
        free(C);
    }
    
    return 0;
}