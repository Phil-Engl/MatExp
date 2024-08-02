#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <assert.h>

#include <cblas.h>
#include "matrix_exponential_flop_count.h"

#define DEBUG 0


#define USEDGEMM 1



#ifndef EIGEN_WRAPPER_H
#define EIGEN_WRAPPER_H
// external call for c++ library
#ifndef __cplusplus
extern "C"{

    #endif

    // Include the Eigen library
    #include <Eigen/Dense>
    #include <Eigen/Eigenvalues>
    #include <Eigen/Core>
    #include <unsupported/Eigen/MatrixFunctions>
    #include <Eigen/LU>


    extern void Eigen_LU_decomposition(const double* inp_mat, const double * RHS, int size, double *out_mat);
    extern void Eigen_MatPow(const double * inp_mat, long power, int n, double * out_mat);



    #ifndef __cplusplus
}
#endif
#endif


const double pade_coefs[14] =
{
    1.0,
    0.5,
    0.12,
    1.833333333333333333333e-2,
    1.992753623188405797101e-3,
    1.630434782608695652174e-4,
    1.035196687370600414079e-5,
    5.175983436853002070393e-7,
    2.043151356652500817261e-8,
    6.306022705717595115002e-10,
    1.483770048404140027059e-11,
    2.529153491597965955215e-13,
    2.810170546219962172461e-15,
    1.544049750670308885967e-17
};

const double theta[15] =
{
    0.0, 
    0.0, 
    0.0,
    1.495585217958292e-2, // theta_3
    0.0,
    2.539398330063230e-1, // theta_5
    0.0,
    9.504178996162932e-1, // theta_7
    0,
    2.097847961257068e0,  // theta_9
    0.0,
    0.0,
    0.0,
    4.25,                 // theta_13 for alg 5.1
    5.371920351148152e0  // theta_13 for alg 3.1
};

const double theta3_exp_10 = 10.150682505756677;
const double theta5_exp_10 = 12.890942410094057;
const double theta7_exp_10 = 25.867904522060577;
const double theta9_exp_10 = 81.486148948487397;
const double theta13_exp_10 = 4416.640977841335371;
const double theta13_inv = 0.235294117647059;

const double coeffs[14] =
{
    0.0,
    0.0,
    0.0,
    2.22819456055355957923e-16, //m = 3
    0.0,
    5.48340615485359898292e-29, //m = 5
    0.0,
    7.84002375459276263332e-43, //m = 7
    0.0,
    1.38702310562852367159e-57, //m = 9
    0.0,
    0.0,
    0.0,
    4.04545010551426617063e-89 //m = 13
    
};

const int64_t ur_int = 0x3ca0000000000000; //hex representation of 2^-53
const double *unit_roundoff = (double*)&ur_int; //2^-53

/**
 * @brief Calculates the matrix multiplication of matrices A and B
 * 
 * @param A The input matrix A with dimensions m x n
 * @param B The input matrix B with dimensions n x t
 * @param C The output matrix C with dimensions m x t
 * @param n common dimension of A and B to multiplicate over
 * @param m number of rows of A and C
 * @param t number of columns of B and C
 */
void mmm(const double *A, const double *B, double *C, int n, int m, int t);

/**
 * @brief copies the matrix A into B
 * 
 * @param A the input matrix A
 * @param B the output matrix B
 * @param m the number of rows of A and B
 * @param n the number of columns of A and B
 */
void copy_matrix(const double* A, double* B, int m, int n);

/**
 * @brief Calculates A to the power of b by repeated squaring (P = A^b)
 * 
 * @param A The input matrix A of dimension n x n
 * @param n The number of rows and columns of A
 * @param b The exponent
 * @param P The output matrix P of dimension n x n
 */
void matpow_by_squaring(const double *A, int n, long b, double *P);
/**
 * @brief Calculates the matrix addition alpha*A + B
 * 
 * @param alpha The scalar multiplier of matrix A
 * @param A The input matrix A with dimensions m x n
 * @param B The input matrix B with dimensions m x n
 * @param C The output matrix C with dimensions m x n
 * @param m number of rows of A, B and C
 * @param n number of columns of A, B and C
 */
void mm_add(double alpha, const double *A, const double *B, double *C, int m, int n);
    
/**     
 * @brief calculates the product alpha * A
 * 
 * @param alpha The scalar multiplier
 * @param A The input matrix A with dimensions m x n
 * @param C The output matrix C with dimensions m x n
 * @param m number of rows of A and C
 * @param n number of columns of A and C
 */
void scalar_matrix_mult(double alpha, const double *A, double *C, int m, int n);

/**
 * @brief returns the absolute values of matrix A
 * 
 * @param A the input matrix
 * @param B the output matrix
 * @param m number of rows of A and B
 * @param n number of columns of A and B
 */
void mat_abs(const double* A, double* B, int m, int n);
/**
 * @brief checks if the matrix A is lower triangular
 * 
 * @param A The input matrix A with dimensions m x n
 * @param m number of rows of A
 * @param n number of columns of A 
 * @return 1 if A is lower triangular
 * @return 0 if A is not lower triangular
 */
int is_lower_triangular(const double *A, int m, int n);

/**
 * @brief checks if the matrix A is upper triangular
 * 
 * @param A The input matrix A with dimensions m x n
 * @param m number of rows of A
 * @param n number of columns of A 
 * @return 1 if A is upper triangular
 * @return 0 if A is not upper triangular
 */
int is_upper_triangular(const double *A, int m, int n);

/**
 * @brief checks if the matrix A has less than m*sqrt(n) non-zero entries
 * 
 * @param A The input matrix A with dimensions m x n
 * @param m the number of rows of A
 * @param n the number of columns of A
 * @return 1 if number of non-zero entries is less than m*sqrt(n) and 0 otherwise
*/

int is_sparse(const double *A, int m, int n);

/**
 * @brief checks if the matrix A is either upper or lower triangular
 * 
 * @param A The input matrix A with dimensions m x n
 * @param m number of rows of A
 * @param n number of columns of A 
 * @return true if A is triangular
 * @return false if A is not triangular
 */
int is_triangular(const double *A, int m, int n);

/**
 * @brief Saves the transpose of A in B
 * 
 * @param A The input matrix A with dimensions m x n 
 * @param B The output matrix B with dimensions n x m
 * @param m number of rows of A
 * @param n number of rows of B
 */
void transpose(const double *A, double *B, int m, int n);

/**
 * @brief calculates the dot product of two vectors
 * 
 * @param v first input vector
 * @param w second input vector
 * @param n length of the vector
 * @return double dot product of the two vectors
 */
double dot_product(const double *v, const double *w, int n);

/**
 * @brief pretty prints a matrix
 * 
 * @param A the input matrix
 * @param m number of rows of the matrix A
 * @param n number of columns of the matrix A
 */
void printmatrix(const double *A, int m, int n);

/**
* @brief Calculates the determinant of a square matrix
*
*@param A the input matrix
*@param n the number of rows and coloumns of matrix A
*@return the determinant of the matrix
*/
double det(const double *A, int n);

/**
 * @brief calculates the adjoint matrix of a square matrix A
 * @param A the input matrix n x n
 * @param B the resulting adjoint matrix n x n
 * @param n the number of rows and coloumns of matrix A
 * @return the adjoint matrix of A

*/
void adj_matrix(const double *A, double *B, int n);


/**
 * @brief calculates the inverse of a square matrix
 * @param A the input matrix n x n
 * @param B the resulting inverse matrix n x n
 * @param n the nuber of rows and coloumns in the matrix
 * @return the inverse of matrix A

*/
void inverse_matrix(const double* A, double* B, int n);

/**
 * @brief calculates the infinity matrix norm of A
 * 
 * @param A the input matrix A of dimension m x n
 * @param m the number of rows of matrix A
 * @param n the number of columns of matrix A
 * @return the infinity matrix norm of A
 */
double infinity_norm(const double* A, int m, int n);

/**
 * @brief fills matrix A with 0s and the diagonal with diag
 * 
 * @param A the input and output matrix A
 * @param diag the value of the diagonal elements in the output
 * @param n the number of rows and columns of matrix A 
 */
void fill_diagonal_matrix(double* A, double diag, int n);


/**
 * @brief computes the sum of the columns of A and stores it in out
 * 
 * @param A the input matrix A of dimension n x n
 * @param n the number of rows/columns of A and the size of the vector out
 * @param out the output vector of size n
 */
void mat_col_sum(const double* A, int n, double* out);


/**
 * @brief computes sinh(x) / x
 * 
 * @param x the input value
 * @return sinh(x) / x
*/
double sinch(double x);

/**
 * @brief computes the matrix exponential of a 2x2 matrix
 * 
 * @param A the input matrix A of dimension 2 x 2
 * @param dest the output matrix of dimension 2 x 2
*/

void expm_2x2_full(const double *A, double*dest);

/**
 * @brief computes the matrix exponential of an upper triangular 2x2 matrix (=> A[2] == 0)
 * 
 * @param A the upper triangular input matrix A of dimension 2 x 2
 * @param dest the output matrix of dimension 2 x 2
*/

void expm_2x2_triangular(const double *A, double*dest);

/**
 * @brief performs forward substitution Ay = b
 * @param A the lower triangular matrix A
 * @param y the destination matrix for the solution y
 * @param b matrix b
 * @param n size of the matrix A, y and b

*/
void forward_substitution_matrix(double * A, double *y, double * b, int n);
void forward_substitution_matrix_vectorized(double * L, double *y, double * b, int n);


void backward_substitution_matrix(double * U, double *x, double * y, int n);
void backward_substitution_matrix_vectorized(double * U, double *x, double * y, int n);

/**
 * @brief computes the LUP decompostion of matrix A with partial pivoting
 * @param org_A the input matrix A
 * @param L destination to store L
 * @param U destitnation to store U
 * @param P destination to store P
*/
void part_piv_lu(double * org_A, double *L, double *U, double *P, int n);

/**
 * @brief solves the linear system of equations Q_m * R_m = P_m by computing the invers of Q_m and multiplying P_m with it
 * @param Q_m LHS matrix Q_m
 * @param P_m RHS matrix P_m
 * @param R_m the destination for the solution
*/
void INV_solve(double* Q_m, double *R_m, double *P_m, int n);

/**
 * @brief solves the linear system of equations Q_m * R_m = P_m by computing the LUP decomposition of Q_m and using forward backward substitution to solve the system
 * @param Q_m LHS matrix Q_m
 * @param P_m RHS matrix P_m
 * @param R_m the destination for the solution
*/
void LU_solve1(double * org_A, double* R_m, double *P_m, int n);

/**
 * @brief solves the linear system of equations Q_m * R_m = P_m by computing the LUP decomposition of Q_m and using forward backward substitution to solve the system
 * (This version uses less space since we combine L and U in one matrix)
 * @param Q_m LHS matrix Q_m
 * @param P_m RHS matrix P_m
 * @param R_m the destination for the solution
*/
void LU_solve2(double * org_A, double* R_m, double *P_m, int n);
void LU_solve2_vec1(double * org_A, double * R_m, double *P_m, int n);



/**
 * @brief solves the linear system of equations Q_m * R_m = P_m by computing the LU decomposition of Q_m in a blocked fashion
 * -> The block size can be set in the function itself, currently set to 1 since this implementation only works if blocksize is divider of n
 * @param Q_m LHS matrix Q_m
 * @param P_m RHS matrix P_m
 * @param R_m the destination for the solution
*/
void Blocked_LU_solve(double * Q_m, double *R_m, double *P_m, int n);



void get_LU_from_eigen(double *input_matrix, double * RHS, int size, double *output_matrix);
void get_MatPow_from_eigen(double *input_matrix, long power, int n, double *output_matrix);




void LU_ColMaj_PivRows_BLAS_vec(double * org_A, double *LU, double *P, int n);//, double *dest);
//void LU_ColMaj_PivRows_vec(double * org_A, double *LU, double *P, int n );//,double *dest){
void LU_ColMaj_PivRows_vec(double *LU, double *P, int n );//,double *dest){


void forward_substitution_ColMaj_LU(double * U, double *x, double * b, int n);
void forward_substitution_ColMaj_vec_kij(double * U, double *x, double * b, int n);
void forward_substitution_ColMaj_vec_ikj(double * U, double *x, double * b, int n);
void forward_substitution_ColMaj_ikj(double * U, double *x, double * b, int n);
void forward_substitution_ColMaj_kij(double * U, double *x, double * b, int n);

void backward_substitution_ColMaj_vec_kij(double * A, double *y, double * b, int n);
void backward_substitution_ColMaj_vec_ikj(double * A, double *y, double * b, int n);
void backward_substitution_ColMaj_kij(double * A, double *y, double * b, int n);
void backward_substitution_ColMaj_ikj(double * A, double *y, double * b, int n);


//used to sort h with stdlib qsort function and keep track of indices
struct h_tmp
{
    double val;
    int idx;
};


const double alpha = 1.0;
const double beta = 0.0;
const CBLAS_LAYOUT layout = CblasColMajor;

long flop_count = 0;

void inc_flop(long c){
    flop_count += c;
}

long get_flop_count(){
    return flop_count;
}

void reset_flop_count(){
    flop_count = 0;
}

void Eigen_LU_decomposition(const double * inp_mat, const double * RHS_in,  int size, double * out_mat){
    Eigen::Map<const Eigen::MatrixXd> A(inp_mat, size, size);
    Eigen::Map<const Eigen::MatrixXd> RHS(RHS_in, size, size);
    Eigen::Map<Eigen::MatrixXd> LU(out_mat, size, size);
    //LU = A.partialPivLu().solve(RHS);
    LU = A.fullPivLu().solve(RHS);  
}

void Eigen_MatPow(const double * inp_mat, long power, int n, double * out_mat){
    Eigen::Map<const Eigen::MatrixXd> A(inp_mat, n, n);
    //Eigen::Map<const Eigen::MatrixXd> RHS(RHS_in, size, size);
    Eigen::Map<Eigen::MatrixXd> PowMat(out_mat, n, n);
    //LU = A.partialPivLu().solve(RHS);
    //LU = A.fullPivLu().solve(RHS); 
    PowMat = A.pow(power);
}

// reference: https://www.ibm.com/docs/en/zos/2.4.0?topic=uatlasal-examples-compiling-linking-running-simple-matrix-multiplication-atlas-program
//TODO USEDGEMM as compile flag
void mmm(const double *A, const double *B, double *C, int n, int m, int t){
    FLOP_COUNT_INC(2*n*m*t, "mmm");
    if(USEDGEMM){
        const double alpha = 1.0;
        const double beta = 0.0;
        CBLAS_LAYOUT layout = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        cblas_dgemm(layout, transA, transB, m, t, n, alpha, A, n, B, t, beta, C, t);
    }else{
        for(int i = 0; i < m; i++){
            for(int j = 0; j < t; j++){
                C[i * t + j] = 0.0;
                for(int k = 0; k < n; k++){
                    C[i * t + j] += A[i * n + k] * B[k * t + j];
                }
            }
        }
    }
}

void copy_matrix(const double* A, double* B, int m, int n){
    // FLOP_COUNT_INC(0, "copy_matrix");
    memcpy(B, A, m*n*sizeof(double));
}


void matpow_by_squaring(const double *A, int n, long b, double *P){
    // FLOP_COUNT_INC(0, "matpow_by_squaring");
    if(b == 1){
        for(int i = 0; i < n*n; i++){
            P[i] = A[i];
        }
        return;
    }

    //P = I
    if(b == 0){
        for(int i = 0; i<n*n; i++){
            P[i] = 0.0;
        }

        for(int i=0; i<n; i++){
            P[i * n + i] = 1.0;  
        }
        return;
    }

    double *curr_P = (double*) malloc(n*n*sizeof(double));
    double *tmp_P = (double*) malloc(n*n*sizeof(double));
    double *cpy_A = (double*) malloc(n*n*sizeof(double));
    double *tmp_A = (double*) malloc(n*n*sizeof(double));
    double *tmp_pointer;
    
    //create a copy of A
    copy_matrix(A,cpy_A,n,n);

    //set curr_P = I
    for(int i = 0; i<n*n; i++){
        curr_P[i] = 0.0;
    }

    for(int i=0; i<n; i++){
        curr_P[i * n + i] = 1.0;  
    }
    
    while(b){
        if(b&1){
            mmm(curr_P, cpy_A, tmp_P, n, n, n);
            tmp_pointer = curr_P;
            curr_P = tmp_P;
            tmp_P = tmp_pointer;   
        }
        b = b >> 1;
        if(!b){
            break;
        }
        mmm(cpy_A, cpy_A, tmp_A, n, n, n);
        tmp_pointer = cpy_A;
        cpy_A = tmp_A;
        tmp_A = tmp_pointer;
    }

    copy_matrix(curr_P, P, n, n);

    free(curr_P);
    free(tmp_P);
    free(cpy_A);
    free(tmp_A);
    return;
}

void mm_add(double alpha, const double *A, const double *B, double *C, int m, int n){
    if(alpha == 0 || alpha == 1.0){
        FLOP_COUNT_INC(n*m, "mm_add");
    }else{
        FLOP_COUNT_INC(2*n*m, "mm_add");
    }
    
    for(int i = 0; i < m*n; i++){
        C[i] = alpha * A[i] + B[i];
    }
}

void scalar_matrix_mult(double alpha, const double *A, double *C, int m, int n){
    FLOP_COUNT_INC(n*m, "scalar_matrix_mult");
    for(int i = 0; i < m*n; i++){
        C[i] = alpha * A[i];
    }
}

void mat_abs(const double* A, double* B, int m, int n){
    FLOP_COUNT_INC(n*m, "mat_abs");
    for(int i = 0; i < m*n; i++){
        B[i] = fabs(A[i]);
    }
}


int is_lower_triangular(const double *A, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = i + 1; j < n; j++){
            FLOP_COUNT_INC(1, "is_lower_triangular");
            if(A[i * n + j] != 0.0) return 0;
        }
    }
    return 1;
}

int is_upper_triangular(const double *A, int m, int n){
    for(int i = 1; i < m; i++){
        for(int j = 0; j < i; j++){
            FLOP_COUNT_INC(1, "is_upper_triangular");
            if(A[i * n + j] != 0.0) return 0;
        }
    }
    return 1;
}

int is_sparse(const double * A, int m, int n){
    int count = 0;
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            if(A[i*n+j] != 0){
                count++;
            }
        }
    }

    
    return (count < m * sqrt(n));
}

int is_triangular(const double *A, int m, int n){
    // FLOP_COUNT_INC(0, "is_triangular");
   if(is_upper_triangular(A, m, n)){
    return 1;
   }else if(is_lower_triangular(A, m, n)){
    return 2;
   }else{
    return 0;
   }
}

void transpose(const double *A, double *B, int m, int n){
    // FLOP_COUNT_INC(0, "transpose");
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            B[j * m + i] = A[i * n + j];
        }
    }
}

double dot_product(const double *v, const double *w, int n){
    FLOP_COUNT_INC(2*n, "dot_product");
    double dp = 0.0;
    for(int i = 0; i < n; i++){
        dp += v[i] * w[i];
    }
    return dp;
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


double infinity_norm(const double* A, int m, int n){
    FLOP_COUNT_INC(2*n*m+m, "infinity_norm");
    double max_row_sum = 0.0;

    for(int i = 0; i < m; i++){
        double curr_row_sum = 0.0;
        for(int j = 0; j < n; j++){
            curr_row_sum += fabs(A[i * n + j]);
        }
        if(curr_row_sum > max_row_sum){
            max_row_sum = curr_row_sum;
        }
    }
    return max_row_sum;
}

void fill_diagonal_matrix(double* A, double diag, int n){
    // FLOP_COUNT_INC(0, "fill_diagonal_matrix");
    for(int i = 0; i < n*n; i++){
        A[i] = 0.0;
    }
    for(int i = 0; i < n; i++){
        A[i * n + i] = diag;
    }
}

void mat_col_sum(const double* A, int n, double* out){
    FLOP_COUNT_INC(n*n, "mat_col_sum");
    for(int j = 0; j < n; j++){
        double curr_sum = 0.0;
        for(int i = 0; i < n; i++){
            curr_sum += A[i * n + j];
        }
        out[j] = curr_sum;
    }
}


void forward_substitution_matrix(double * A, double *y, double * b, int n){
    for(int col=0; col<n; col++){
        for(int i=0; i<n; i++){
            y[i*n + col] = b[i*n + col];
            for(int j=0; j<i; j++){
                FLOP_COUNT_INC(2, "forward_substitution_matrix");
                y[i*n+col] = y[i*n + col] - A[i*n+j] * y[j*n + col];
            }
            FLOP_COUNT_INC(1,"forward_substitution_matrix");
            y[i*n+col] = y[i*n + col] / A[i*n+i];
        }
    }
}
/*
void forward_substitution_matrix_vectorized(double * L, double *y, double * b, int n){
   for(int col=0; col<n; col+=4){

    for(int i=0; i<n; i++){

        __m256d res = _mm256_loadu_pd( &b[i*n+col] );
        __m256d divisor = _mm256_set1_pd( L[i*n+i] );

        for(int j=0; j<i; j++){
             __m256d L_vec = _mm256_set1_pd( L[i*n+j] );
             __m256d y_vec = _mm256_loadu_pd( &y[j*n+col] );
             __m256d prod = _mm256_mul_pd(L_vec, y_vec);
             res = _mm256_sub_pd(res, prod);


        }
        
            res = _mm256_div_pd(res, divisor);
            _mm256_storeu_pd(&y[i*n+col], res);

    }

}
}*/

void backward_substitution_matrix(double * U, double *x, double * y, int n){
    for(int col=0; col<n; col++){
        for(int i=n-1; i>=0; i--){
            FLOP_COUNT_INC(1, "backward_substitution_matrix");
            x[i*n + col] = y[i*n + col] / U[i*n+i];
            for(int j=i+1; j<n; j++){
                FLOP_COUNT_INC(3, "backward_substitution_matrix");
                x[i*n + col] = x[i * n+col] - (U[i*n+j] * x[j*n+col]) / U[i*n+i];
            }
            //x[i*n + col] = x[i*n + col] / U[i*n + i];
        }

    }
}

/*
void backward_substitution_matrix_vectorized(double * U, double *x, double * y, int n){
    for(int col=0; col<n; col+=4){

        for(int i=n-1; i>=0; i--){

            __m256d res = _mm256_loadu_pd( &y[i*n+col] );
            __m256d divisor = _mm256_set1_pd( U[i*n+i] );

            for(int j=i+1; j<n; j++){
                __m256d U_vec = _mm256_set1_pd( U[i*n+j] );
                __m256d x_vec = _mm256_loadu_pd( &x[j*n+col] );
                __m256d prod = _mm256_mul_pd(U_vec, x_vec);
                res = _mm256_sub_pd(res, prod);
            }

            res = _mm256_div_pd(res, divisor);
            _mm256_storeu_pd(&x[i*n+col], res);

        }

    }
}
*/

/*
void part_piv_lu(double * org_A, double *L, double *U, double *P, int n){
    // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    double curr_piv;
    int index_piv;
    double tmp;

    double *A = (double*) malloc(n*n*sizeof(double));

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = org_A[i*n+j];
            if(i==j){
                L[i*n+j] = 1;
                U[i*n+j] = 1;
            }else{
                L[i*n+j] = 0;
                U[i*n+j] = 0;
            }
        }
    }

    int *tmp_P = (int*) malloc(n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = A[k*n+k];
        index_piv = k;
        
        for(int i=k+1; i<n; i++){ // or just j????  
            FLOP_COUNT_INC(3, "part_piv_lu");
            if( fabs(A[i*n+k]) > fabs(curr_piv) ){
                curr_piv = A[i*n+k];
                index_piv = i;
            }
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
            for(int x=k; x<n; x++){
                tmp = A[k*n + x];
                A[k*n +x] = A[index_piv*n+x];
                A[index_piv*n+x] = tmp;
            }

            
            for(int y=0; y<k; y++){
                tmp = L[k*n+y];
                L[k*n+y] = L[index_piv*n+y];
                L[index_piv*n+y] = tmp;
            }

            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

    
        for(int j=k; j<n; j++){
            U[k*n+j] = A[k*n+j];

            FLOP_COUNT_INC(2*k, "part_piv_lu");
            for(int s=0; s<k; s++){
                U[k*n+j] -= L[k*n+s] * U[s*n+j]; 
            }
        }

    
        for(int i=k+1; i<n; i++){
            L[i*n+k] = A[i*n+k];// / U[k*n+k];
            FLOP_COUNT_INC(2*k, "part_piv_lu");
            for(int s=0; s<k; s++){
                L[i*n+k] -= (L[i*n+s]*U[s*n+k]);// / U[k*n+k];
            }
            FLOP_COUNT_INC(1, "part_piv_lu");
            L[i*n+k] /= U[k*n+k];
        }

    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[i*n + tmp_P[i]] = 1.0;
    }
    
    free(tmp_P);
    free(A);
}
*/
void LU_solve1(double * Q_m, double * R_m, double *P_m, int n){
    double *L = (double*) malloc(n*n*sizeof(double));
    double *U = (double*) malloc(n*n*sizeof(double));
    double *P = (double*) malloc(n*n*sizeof(double));
    double *Y = (double*) malloc(n*n*sizeof(double));
    double *A = (double*) malloc(n*n*sizeof(double));
    int *tmp_P = (int*) malloc(n*sizeof(int));
    double *permuted_P_m = (double*) malloc(n*n*sizeof(double));

    double curr_piv;
    int index_piv;
    double tmp;


    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = Q_m[i*n+j];
            if(i==j){
                L[i*n+j] = 1;
                U[i*n+j] = 1;
            }else{
                L[i*n+j] = 0;
                U[i*n+j] = 0;
            }
        }
    }

   
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = A[k*n+k];
        index_piv = k;

        for(int i=k+1; i<n; i++){ // or just j????  
            FLOP_COUNT_INC(3, "LU_solve1");
            if( fabs(A[i*n+k]) > fabs(curr_piv) ){
                curr_piv = A[i*n+k];
                index_piv = i;
            }
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
            for(int x=k; x<n; x++){
                tmp = A[k*n + x];
                A[k*n +x] = A[index_piv*n+x];
                A[index_piv*n+x] = tmp;
            }

            
            for(int y=0; y<k; y++){
                tmp = L[k*n+y];
                L[k*n+y] = L[index_piv*n+y];
                L[index_piv*n+y] = tmp;
            }

            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

    
        for(int j=k; j<n; j++){
            U[k*n+j] = A[k*n+j];

            FLOP_COUNT_INC(2*k, "LU_solve1");
            for(int s=0; s<k; s++){
                U[k*n+j] -= L[k*n+s] * U[s*n+j]; 
            }
        }

    
        for(int i=k+1; i<n; i++){
            L[i*n+k] = A[i*n+k];// / U[k*n+k];
            FLOP_COUNT_INC(2*k, "LU_solve1");
            for(int s=0; s<k; s++){
                L[i*n+k] -= (L[i*n+s]*U[s*n+k]);// / U[k*n+k];
            }
            FLOP_COUNT_INC(1, "LU_solve1");
            L[i*n+k] /= U[k*n+k];
        }

    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[i*n + tmp_P[i]] = 1.0;
    }
    
    
    mmm(P, P_m, permuted_P_m, n, n, n);


    for(int col=0; col<n; col++){
        for(int i=0; i<n; i++){
            FLOP_COUNT_INC(1, "LU_solve1");
            Y[i*n + col] = permuted_P_m[i*n + col] / L[i*n+i];
            for(int j=0; j<i; j++){
                FLOP_COUNT_INC(3, "LU_solve1");
                Y[i*n+col] -= (L[i*n+j] * Y[j*n + col]) / L[i*n+i];
            }
        }
    }


    for(int col=0; col<n; col++){
        for(int i=n-1; i>=0; i--){
            FLOP_COUNT_INC(1, "LU_solve1");
            R_m[i*n + col] = Y[i*n + col] / U[i*n+i];
            for(int j=i+1; j<n; j++){
                FLOP_COUNT_INC(3, "LU_solve1");
                R_m[i*n + col] -= (U[i*n+j] * R_m[j*n+col]) / U[i*n+i];
            }
        }
    }


    free(tmp_P);
    free(A);
    free(L);
    free(U);
    free(P);
    free(Y);
    free(permuted_P_m);
}
/**
 * @brief This is a C implementation of algorithm 2.4 from the
 * 2000 Higham and Tisseur paper.
 * It is made with the help of the scipy implementation
 * https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_onenormest.py
 * 
 */

//TODO: rewrite for more efficient column major access 

/* ----- helper functions ----- */

/**
 * @brief checks if all columns in A are parallel to any column of B
 * 
 * @param A the input matrix A (transposed)
 * @param B the input matrix B (transposed)
 * @param m number of rows of A and B
 * @param n number of columns of A and B
 * @return int 1, if the condition in the description is met, 0 otherwise.
 */
int check_all_columns_parallel(const double *A, const double *B, int m, int n){
    for(int i = 0; i < n; i++){
        int flag = 0;
        for(int j = 0; j < n; j++){
            FLOP_COUNT_INC(1, "check_all_columns_parallel");
            flag = flag || (dot_product(&A[i * m], &B[j * m], m) == (double)m);
        }
        if(!flag){    
            return 0;
        }
    }
    return 1;
}

/**
 * @brief column in A needs resampling if it is parallel to any of its previous columns
 * or any column in B
 * 
 * @param k the index of the current row
 * @param v the current row
 * @param A the matrix A in which the current row resides
 * @param B the matrix B
 * @param m number of rows of A and B
 * @param n number of columns of A and B
 * @return int 1 if the the row needs resampling, 0 otherwise 
 */
int column_needs_resampling(int k, const double* v, const double* A, const double* B, int m, int n){
    //Assume all entries in the matrices are 1.0 or -1.0.
    //Comparison with double casted int would be dangerous otherwise
    for(int i = 0; i < k; i++){
        FLOP_COUNT_INC(1, "column_needs_resampling");
        if(dot_product(v, &A[i * n], n) == (double)n){
            return 1;
        }
    }
    for(int i = 0; i < m; i++){
        FLOP_COUNT_INC(1, "column_needs_resampling");
        if(dot_product(v, &B[i * n], n) == (double)n){
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Ensures that no column of A is parallel to another column of A
 * or to a column of B by replacing columns of A by rand{−1, 1}.
 * 
 * @param A Input matrix A, with all entries in {-1, 1} (transposed)
 * @param B Input matrix B, with all entries in {-1, 1} (transposed)
 * @param m number of rows of A and B 
 * @param n number of columns of A and B
 */
void resample_columns(double *A, double *B, int m, int n){
    // TODO: FLOP_COUNT_INC(0, "resample_columns"); 
    for(int i = 0; i < n; i++){
        while(column_needs_resampling(i, &A[i * m], A, B, n, m)){
            for(int j = 0; j < m; j++){
                A[i * m + j] = (double)(((rand() % 2) * 2) - 1); 
            }
        }
    }
}

/**
 * @brief checks if an index is already in the index history
 * 
 * @param idx the index to check
 * @param hist the index history 
 * @param hist_len the length of the index history
 * @return int 1 if the index is in the history, 0 otherwise
 */
int idx_in_hist(int idx, int *hist, int hist_len){
    // FLOP_COUNT_INC(0, "idx_in_hist");
    int flag = 0;
    for(int i = 0; i < hist_len; i++){
        flag = flag || (idx == hist[i]);
    }
    return flag;
}

/**
 * @brief compare function for qsort
 * 
 * @param a struct h_tmp* a
 * @param b struct h_tmp* b
 * @return int -1 if (*a).val > (*b).val, -1 if (*a).val < (*b).val, 0 otherwise.
 */
int cmp_h(const void *a, const void *b){
    struct h_tmp* a1 = (struct h_tmp*)a;
    struct h_tmp* b1 = (struct h_tmp*)b;
    FLOP_COUNT_INC(1, "cmp_h");
    if((*a1).val > (*b1).val){
        return -1;
    } else if ((*a1).val < (*b1).val){
        FLOP_COUNT_INC(1, "cmp_h");
        return 1;
    } else {
        return 0;
    }
}

/* ----- one norm functions ----- */

double onenorm_best(const double* A, int m, int n, int* max_idx){
    FLOP_COUNT_INC(2*n*m, "onenorm_best");
    double max = 0.0;
    for(int j = 0; j < n; j++){
        double curr = 0.0;
        for(int i = 0; i < m; i++){
            curr += fabs(A[i * n + j]);
        }
        if(curr > max){
            max = curr;
            *max_idx = j;
        }
    }
    return max;
}

double onenorm(const double* A, int m, int n){
    // FLOP_COUNT_INC(0, "onenorm"); Note: all flops are already accounted for in onenorm_best
    int best = 0;
    return onenorm_best(A,m,n,&best);
}

double onenormest(const double* A, int n, int t, int itmax, int get_v, int get_w, double* v, double* w){
    // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    if(t >= n || t < 1 || itmax < 2){
        return -1.0;
    }
    
    int k = 1;
    int best_j = 0;
    int ind_best = 0;
    int hist_len = 0;
    int new_length = t;

    double est = 0.0;
    double est_old = 0.0;
    
    double max_h = 0.0;
    FLOP_COUNT_INC(2, "onenormest");
    double x_elem = 1.0 / (double)n;
    double x_elem_n = -1.0 * x_elem;
    
    int* ind_hist = (int*)malloc(n * itmax * sizeof(int));
    int* ind = (int*)malloc(n * sizeof(int));
    int* ind_in = (int*)malloc(n * sizeof(int));
    
    double* AT = (double*)malloc(n * n * sizeof(double));
    double* S = (double*)malloc(n * t * sizeof(double));
    double* S_old = (double*)malloc(n * t * sizeof(double));
    double *S_T = (double*)malloc(n * t * sizeof(double));
    double *S_old_T = (double*)malloc(n * t * sizeof(double));
    double* X = (double*)malloc(n * t * sizeof(double));
    double* Y = (double*)malloc(n * t * sizeof(double));
    double* Z = (double*)malloc(n * t * sizeof(double));
    
    struct h_tmp* h = (struct h_tmp*)malloc(n * sizeof(struct h_tmp));

    srand(time(0));

    //initialize matrices and vectors
    //X is similar to the scipy implementation, just more efficiently built
    for(int i = 0; i < n; i++){
        ind_hist[i] = -1;
        ind[i] = -1;
        for(int j = 0; j < t; j++){
            S[i * t + j] = 0.0;
            if(j == 0 || i >= j){
                X[i * t + j] = x_elem;
            } else {
                X[i * t + j] = x_elem_n;
            }
        }
    }
    transpose(A, AT, n, n);

    while(1){
        //Y = A * X
        mmm(A, X, Y, n, n, t);
        est = onenorm_best(Y,n,t, &best_j); 
        FLOP_COUNT_INC(1, "onenormest");
        if(est > est_old || k == 2){
            if(k >= 2){
                ind_best = ind[best_j];
            }
            if(get_w){
                for(int i = 0; i < n; i++){
                    w[i] = Y[i * t + best_j];
                }
            }
        }
        //(1)
        if(k >= 2 && est <= est_old){
            est = est_old;
            break;
        }
        est_old = est;
        double *S_temp = S_old;
        S_old = S;
        S = S_temp;


        if(k > itmax){
            break;
        }

        //S = sign(Y)
        FLOP_COUNT_INC(n*t, "onenormest");            
        for(int i = 0; i < n * t; i++){
           S[i] = Y[i] >= 0.0 ? 1.0 : -1.0;        
        }
        
        //(2)
        //transpose the matrices for better access
        transpose(S, S_T, n, t);
        transpose(S_old, S_old_T, n, t);
        //If every column of S is parallel to a column of S_old, break 
        if(check_all_columns_parallel(S_T, S_old_T, n, t)){
            break;
        }

        /* Ensure that no column of S is parallel to another column of S
        or to a column of S_old by replacing columns of S by rand{−1, 1}. */
        resample_columns(S_T, S_old_T, n, t);
        transpose(S_T, S, t, n);

        //(3)
        mmm(AT, S, Z, n, n, t);

        max_h = 0.0;
        FLOP_COUNT_INC(2*n*t, "onenormest");
        for(int i = 0; i < n; i++){
            h[i].val = 0.0;
            h[i].idx = i;
            for(int j = 0; j < t; j++){
                double a = fabs(Z[i * t + j]);
                if(a > h[i].val){
                    h[i].val = a;
                    FLOP_COUNT_INC(1, "onenormest");
                    if(a > max_h){
                        max_h = a;
                    }
                }
            }
        }
        
        //(4)
        if(k >= 2 && max_h == h[ind_best].val){
            break;
        }
        
        //Sort h so that h[0] ≥ · · · ≥ h[n-1] and re-order ind correspondingly.
        qsort(h,n, sizeof(struct h_tmp), cmp_h);
        
        
        //(5) 
        /*  
         *  If ind(0:t) is contained in ind_hist, break
         *  Replace ind(0:t) by the first t indices in ind(0:n) that are
         *  not in ind_hist. 
        */
        
        int flag = 1;
        int i_in_hist = 0;
        int outidx = 0;
        int inidx = 0;
        for(int i = 0; i < n; i++){
            i_in_hist = idx_in_hist(h[i].idx, ind_hist, hist_len);
            flag = flag && i_in_hist;
            if(i == t-1 && flag){
                break;
            }
            if(!i_in_hist){
                ind[outidx++] = h[i].idx;
            }else{
                ind_in[inidx++] = h[i].idx;
            }
        }
        
        if(flag){
            break;
        }
        
        new_length = outidx;
        for(int i = 0; i < n; i++){
            if(outidx == n){
                break;
            }
            ind[outidx++] = ind_in[i];
        }
        
        
        /*  
         *  create a new X (n x t) matrix with the unit vectors from 
         *  the best new indices 
         */
        for(int i = 0; i < n * t; i++){
            X[i] = 0.0;
        }

        for(int i = 0; i < t; i++){
            X[ind[i] * t + i] = 1.0;
        }

        //add the new indices to ind_hist
        int nhlen = hist_len + t;
        for(int i = 0; i < nhlen; i++){
            if(i == new_length){
                break;
            }
            ind_hist[hist_len++] = ind[i];
        }
        k++;
    }
    //(6)
    //optionally create v, the unit vector of ind_best
    if(get_v){
        for(int i = 0; i < n; i++){
            v[i] = 0.0;
        }
        v[ind_best] = 1.0;
    }
    
    free(ind_hist);
    free(ind);
    free(ind_in);
    free(AT);
    free(S);
    free(S_old);
    free(S_T);
    free(S_old_T);
    free(X);
    free(Y);
    free(Z);
    free(h);

    return est;
}

double normest(const double* A, int n){
    // FLOP_COUNT_INC(0, "normest"); Note: all flops are already accounted for in onenormest
    double* dummyv = (double*)malloc(sizeof(double));
    double* dummyw = (double*)malloc(sizeof(double));
    int t = 1;
    if(n > 2){
        t = 2;
    }
    
    double res = onenormest(A,n,t,5,0,0,dummyv,dummyw);
   
    free(dummyv);
    free(dummyw);
    return res;
    
}/**
 * @brief eval 3.4 for m=3
 */
void eval3_4_m3(const double* A, const double* A_2, int n, double *P_3, double *Q_3)
{   

    FLOP_COUNT_INC(6*n*n, "eval3_4_m3");
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    // compute u and v separately
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j+=4){
            
            int idx_0  = i*n + j + 0;
            
            int idx_1  = i*n + j + 1;
            
            int idx_2  = i*n + j + 2;
            
            int idx_3  = i*n + j + 3;
            

            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_0] = (i== j+0 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_0];
            V[idx_0] = (i== j+0 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_0];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_1] = (i== j+1 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_1];
            V[idx_1] = (i== j+1 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_1];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_2] = (i== j+2 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_2];
            V[idx_2] = (i== j+2 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_2];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_3] = (i== j+3 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_3];
            V[idx_3] = (i== j+3 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_3];
            
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4){
        
        P_3[i+0] = Temp[i+0] + V[i+0];
        Q_3[i+0] = V[i+0] - Temp[i+0]; // (-A)*U + V == -(A*U) + V
        
        P_3[i+1] = Temp[i+1] + V[i+1];
        Q_3[i+1] = V[i+1] - Temp[i+1]; // (-A)*U + V == -(A*U) + V
        
        P_3[i+2] = Temp[i+2] + V[i+2];
        Q_3[i+2] = V[i+2] - Temp[i+2]; // (-A)*U + V == -(A*U) + V
        
        P_3[i+3] = Temp[i+3] + V[i+3];
        Q_3[i+3] = V[i+3] - Temp[i+3]; // (-A)*U + V == -(A*U) + V
        
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief eval 3.4 for m=5
 */
void eval3_4_m5(const double* A, const double* A_2, const double* A_4, int n, double *P_5, double *Q_5)
{       
    FLOP_COUNT_INC(10*n*n, "eval3_4_m5");
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    // compute u and v separately
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j+=4){
            
            int idx_0  = i*n + j + 0;
            
            int idx_1  = i*n + j + 1;
            
            int idx_2  = i*n + j + 2;
            
            int idx_3  = i*n + j + 3;
            
            
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_0] = (i== j+0 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_0] + pade_coefs[5] * A_4[idx_0];
            V[idx_0] = (i== j+0 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_0] + pade_coefs[4] * A_4[idx_0];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_1] = (i== j+1 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_1] + pade_coefs[5] * A_4[idx_1];
            V[idx_1] = (i== j+1 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_1] + pade_coefs[4] * A_4[idx_1];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_2] = (i== j+2 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_2] + pade_coefs[5] * A_4[idx_2];
            V[idx_2] = (i== j+2 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_2] + pade_coefs[4] * A_4[idx_2];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_3] = (i== j+3 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_3] + pade_coefs[5] * A_4[idx_3];
            V[idx_3] = (i== j+3 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_3] + pade_coefs[4] * A_4[idx_3];
            
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4){
        
        P_5[i+0] = Temp[i+0] + V[i+0];
        Q_5[i+0] = V[i+0] - Temp[i+0]; // (-A)*U + V == -(A*U) + V
        
        P_5[i+1] = Temp[i+1] + V[i+1];
        Q_5[i+1] = V[i+1] - Temp[i+1]; // (-A)*U + V == -(A*U) + V
        
        P_5[i+2] = Temp[i+2] + V[i+2];
        Q_5[i+2] = V[i+2] - Temp[i+2]; // (-A)*U + V == -(A*U) + V
        
        P_5[i+3] = Temp[i+3] + V[i+3];
        Q_5[i+3] = V[i+3] - Temp[i+3]; // (-A)*U + V == -(A*U) + V
        
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief eval 3.4 for m=7
 */
void eval3_4_m7(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, double *P_7, double *Q_7)
{       
    FLOP_COUNT_INC(14*n*n, "eval3_4_m7");
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    // compute u and v separately
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j+=4){
            
            int idx_0  = i*n + j + 0;
            
            int idx_1  = i*n + j + 1;
            
            int idx_2  = i*n + j + 2;
            
            int idx_3  = i*n + j + 3;
            
            
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_0] = (i== j+0 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_0] + pade_coefs[5] * A_4[idx_0] + pade_coefs[7] * A_6[idx_0];
            V[idx_0] = (i== j+0 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_0] + pade_coefs[4] * A_4[idx_0] + pade_coefs[6] * A_6[idx_0];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_1] = (i== j+1 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_1] + pade_coefs[5] * A_4[idx_1] + pade_coefs[7] * A_6[idx_1];
            V[idx_1] = (i== j+1 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_1] + pade_coefs[4] * A_4[idx_1] + pade_coefs[6] * A_6[idx_1];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_2] = (i== j+2 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_2] + pade_coefs[5] * A_4[idx_2] + pade_coefs[7] * A_6[idx_2];
            V[idx_2] = (i== j+2 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_2] + pade_coefs[4] * A_4[idx_2] + pade_coefs[6] * A_6[idx_2];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_3] = (i== j+3 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_3] + pade_coefs[5] * A_4[idx_3] + pade_coefs[7] * A_6[idx_3];
            V[idx_3] = (i== j+3 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_3] + pade_coefs[4] * A_4[idx_3] + pade_coefs[6] * A_6[idx_3];
            
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4){
        
        P_7[i+0] = Temp[i+0] + V[i+0];
        Q_7[i+0] = V[i+0] - Temp[i+0]; // (-A)*U + V == -(A*U) + V
        
        P_7[i+1] = Temp[i+1] + V[i+1];
        Q_7[i+1] = V[i+1] - Temp[i+1]; // (-A)*U + V == -(A*U) + V
        
        P_7[i+2] = Temp[i+2] + V[i+2];
        Q_7[i+2] = V[i+2] - Temp[i+2]; // (-A)*U + V == -(A*U) + V
        
        P_7[i+3] = Temp[i+3] + V[i+3];
        Q_7[i+3] = V[i+3] - Temp[i+3]; // (-A)*U + V == -(A*U) + V
        
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief eval 3.4 for m=9
 */
void eval3_4_m9(const double* A, const double* A_2, const double* A_4, const double* A_6, const double* A_8, int n, double *P_9, double *Q_9)
{       
    FLOP_COUNT_INC(18*n*n, "eval3_4_m9");
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    // compute u and v separately
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j+=4){
            
            int idx_0  = i*n + j + 0;
            
            int idx_1  = i*n + j + 1;
            
            int idx_2  = i*n + j + 2;
            
            int idx_3  = i*n + j + 3;
            
            
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_0] = (i== j+0 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_0] + pade_coefs[5] * A_4[idx_0] 
                            + pade_coefs[7] * A_6[idx_0] + pade_coefs[9] * A_8[idx_0];
            V[idx_0] = (i== j+0 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_0] + pade_coefs[4] * A_4[idx_0] 
                            + pade_coefs[6] * A_6[idx_0] + pade_coefs[8] * A_8[idx_0];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_1] = (i== j+1 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_1] + pade_coefs[5] * A_4[idx_1] 
                            + pade_coefs[7] * A_6[idx_1] + pade_coefs[9] * A_8[idx_1];
            V[idx_1] = (i== j+1 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_1] + pade_coefs[4] * A_4[idx_1] 
                            + pade_coefs[6] * A_6[idx_1] + pade_coefs[8] * A_8[idx_1];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_2] = (i== j+2 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_2] + pade_coefs[5] * A_4[idx_2] 
                            + pade_coefs[7] * A_6[idx_2] + pade_coefs[9] * A_8[idx_2];
            V[idx_2] = (i== j+2 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_2] + pade_coefs[4] * A_4[idx_2] 
                            + pade_coefs[6] * A_6[idx_2] + pade_coefs[8] * A_8[idx_2];
            
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_3] = (i== j+3 ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_3] + pade_coefs[5] * A_4[idx_3] 
                            + pade_coefs[7] * A_6[idx_3] + pade_coefs[9] * A_8[idx_3];
            V[idx_3] = (i== j+3 ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_3] + pade_coefs[4] * A_4[idx_3] 
                            + pade_coefs[6] * A_6[idx_3] + pade_coefs[8] * A_8[idx_3];
            
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4){
        
        P_9[i+0] = Temp[i+0] + V[i+0];
        Q_9[i+0] = V[i+0] - Temp[i+0]; // (-A)*U + V == -(A*U) + V
        
        P_9[i+1] = Temp[i+1] + V[i+1];
        Q_9[i+1] = V[i+1] - Temp[i+1]; // (-A)*U + V == -(A*U) + V
        
        P_9[i+2] = Temp[i+2] + V[i+2];
        Q_9[i+2] = V[i+2] - Temp[i+2]; // (-A)*U + V == -(A*U) + V
        
        P_9[i+3] = Temp[i+3] + V[i+3];
        Q_9[i+3] = V[i+3] - Temp[i+3]; // (-A)*U + V == -(A*U) + V
        
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief Evaluates equation 3.4. Special case for A where powers are not precomputed
 * 
 * @param A_abs The input matrix A with dimension n x n
 * @param n The number of rows and columns of A
 * @param m The padé approximant
 * @param P_m_abs The output matrix p_m(A) with dimension n x n
 * @param Q_m_abs The output matrix q_m(A) with dimension n x n
 */
void eval3_4_abs(const double *A_abs, int n, int m, double *P_m_abs, double *Q_m_abs){
    double *A_abs_2 = (double*) aligned_alloc(32, n*n*sizeof(double));


    mmm(A_abs, A_abs, A_abs_2, n, n, n);

    if(m == 3){
        eval3_4_m3(A_abs, A_abs_2, n, P_m_abs, Q_m_abs);
        free(A_abs_2);
        return;
    }

    double *A_abs_4 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_abs_2, A_abs_2, A_abs_4, n, n, n);

    if(m == 5){
        eval3_4_m5(A_abs, A_abs_2, A_abs_4, n, P_m_abs, Q_m_abs);
        free(A_abs_2);
        free(A_abs_4);
        return;
    }

    double *A_abs_6 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_abs_2, A_abs_4, A_abs_6, n, n, n);

    if(m == 7){
        eval3_4_m7(A_abs, A_abs_2, A_abs_4, A_abs_6, n, P_m_abs, Q_m_abs);
        free(A_abs_2);
        free(A_abs_4);
        free(A_abs_6);
        return;
    }

    double *A_abs_8 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_abs_4, A_abs_4, A_abs_8, n, n, n);
    eval3_4_m9(A_abs, A_abs_2, A_abs_4, A_abs_6, A_abs_8, n, P_m_abs, Q_m_abs);

    free(A_abs_2);
    free(A_abs_4);
    free(A_abs_6);
    free(A_abs_8);
}


/**
 * @brief Evaluates equation 3.5
 * 
 * @param A The input matrix A with dimension n x n
 * @param A_2 to A_8: precomputed powers of A
 * @param n The number of rows and columns of A, A_2, A_4 and A_6
 * @param P_13 The output matrix p_13(A) with dimension n x n
 * @param Q_13 The output matrix q_13(A) with dimension n x n
 */
void eval3_5(const double *A, double* A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13){
    FLOP_COUNT_INC(26*n*n, "eval3_5");
    double *U_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *U_tmp2 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V_tmp2 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j+=4){
            
            int idx_0  = i*n + j + 0;
            
            int idx_1  = i*n + j + 1;
            
            int idx_2  = i*n + j + 2;
            
            int idx_3  = i*n + j + 3;
            

            
            U_tmp[idx_0]  = pade_coefs[13]*A_6[idx_0] + pade_coefs[11]*A_4[idx_0] + pade_coefs[9]*A_2[idx_0];
            V_tmp[idx_0]  = pade_coefs[12]*A_6[idx_0] + pade_coefs[10]*A_4[idx_0] + pade_coefs[8]*A_2[idx_0];
            U_tmp2[idx_0] = pade_coefs[7]*A_6[idx_0] + pade_coefs[5]*A_4[idx_0] + (pade_coefs[3]*A_2[idx_0] + (i== j+0 ? pade_coefs[1] : 0.0));
            V_tmp2[idx_0] = pade_coefs[6]*A_6[idx_0] + pade_coefs[4]*A_4[idx_0] + (pade_coefs[2]*A_2[idx_0] + (i== j+0 ? pade_coefs[0] : 0.0));
            
            U_tmp[idx_1]  = pade_coefs[13]*A_6[idx_1] + pade_coefs[11]*A_4[idx_1] + pade_coefs[9]*A_2[idx_1];
            V_tmp[idx_1]  = pade_coefs[12]*A_6[idx_1] + pade_coefs[10]*A_4[idx_1] + pade_coefs[8]*A_2[idx_1];
            U_tmp2[idx_1] = pade_coefs[7]*A_6[idx_1] + pade_coefs[5]*A_4[idx_1] + (pade_coefs[3]*A_2[idx_1] + (i== j+1 ? pade_coefs[1] : 0.0));
            V_tmp2[idx_1] = pade_coefs[6]*A_6[idx_1] + pade_coefs[4]*A_4[idx_1] + (pade_coefs[2]*A_2[idx_1] + (i== j+1 ? pade_coefs[0] : 0.0));
            
            U_tmp[idx_2]  = pade_coefs[13]*A_6[idx_2] + pade_coefs[11]*A_4[idx_2] + pade_coefs[9]*A_2[idx_2];
            V_tmp[idx_2]  = pade_coefs[12]*A_6[idx_2] + pade_coefs[10]*A_4[idx_2] + pade_coefs[8]*A_2[idx_2];
            U_tmp2[idx_2] = pade_coefs[7]*A_6[idx_2] + pade_coefs[5]*A_4[idx_2] + (pade_coefs[3]*A_2[idx_2] + (i== j+2 ? pade_coefs[1] : 0.0));
            V_tmp2[idx_2] = pade_coefs[6]*A_6[idx_2] + pade_coefs[4]*A_4[idx_2] + (pade_coefs[2]*A_2[idx_2] + (i== j+2 ? pade_coefs[0] : 0.0));
            
            U_tmp[idx_3]  = pade_coefs[13]*A_6[idx_3] + pade_coefs[11]*A_4[idx_3] + pade_coefs[9]*A_2[idx_3];
            V_tmp[idx_3]  = pade_coefs[12]*A_6[idx_3] + pade_coefs[10]*A_4[idx_3] + pade_coefs[8]*A_2[idx_3];
            U_tmp2[idx_3] = pade_coefs[7]*A_6[idx_3] + pade_coefs[5]*A_4[idx_3] + (pade_coefs[3]*A_2[idx_3] + (i== j+3 ? pade_coefs[1] : 0.0));
            V_tmp2[idx_3] = pade_coefs[6]*A_6[idx_3] + pade_coefs[4]*A_4[idx_3] + (pade_coefs[2]*A_2[idx_3] + (i== j+3 ? pade_coefs[0] : 0.0));
            
        }
    }

    mmm(A_6, U_tmp, U, n, n, n);
    mmm(A_6, V_tmp, V, n, n, n);

    mm_add(1.0, U, U_tmp2, U, n, n);
    mm_add(1.0, V, V_tmp2, V, n, n);

    mmm(A, U, U_tmp, n, n, n); 

    for(int i=0; i<n*n; i+=4){
        
        P_13[i+0] = U_tmp[i+0] + V[i+0];
        Q_13[i+0] = V[i+0] - U_tmp[i+0]; // (-A)*U + V == -(A*U) + V
        
        P_13[i+1] = U_tmp[i+1] + V[i+1];
        Q_13[i+1] = V[i+1] - U_tmp[i+1]; // (-A)*U + V == -(A*U) + V
        
        P_13[i+2] = U_tmp[i+2] + V[i+2];
        Q_13[i+2] = V[i+2] - U_tmp[i+2]; // (-A)*U + V == -(A*U) + V
        
        P_13[i+3] = U_tmp[i+3] + V[i+3];
        Q_13[i+3] = V[i+3] - U_tmp[i+3]; // (-A)*U + V == -(A*U) + V
        
    }

    free(U_tmp);
    free(V_tmp);
    free(U_tmp2);
    free(V_tmp2);
    free(U);
    free(V);
}

/**
 * @brief Evaluates equation 3.5. Special case for A where powers are not precomputed
 * 
 * @param A_abs The input matrix A with dimension n x n
 * @param n The number of rows and columns of A
 * @param P_13_abs The output matrix p_13(A) with dimension n x n
 * @param Q_13_abs The output matrix q_13(A) with dimension n x n
 */
void eval3_5_abs(const double *A_abs, int n, double *P_13_abs, double *Q_13_abs){
    double *A_abs_2 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_abs_4 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_abs_6 = (double*) aligned_alloc(32, n*n*sizeof(double));

    mmm(A_abs, A_abs, A_abs_2, n, n, n);
    mmm(A_abs_2, A_abs_2, A_abs_4, n, n, n);
    mmm(A_abs_2, A_abs_4, A_abs_6, n, n, n);

    eval3_5(A_abs, A_abs_2, A_abs_4, A_abs_6, n, P_13_abs, Q_13_abs);

    free(A_abs_2);
    free(A_abs_4);
    free(A_abs_6);
}

/**
 * @brief Evaluates equation 3.6 -> solves the linear system Q_m * R_m = P_m
 * 
 * @param P_m The input matrix p_m(A) of dimension n x n
 * @param Q_m The input matrix q_m(A) of dimension n x n
 * @param n The size of the matrices
 * @param R_m The output matrix r_m(A) of dimension n x n
 */
void eval3_6(double *P_m, double *Q_m, int n, double *R_m, int triangular_indicator){
    // FLOP_COUNT_INC(0, "eval3_6"); Note: all flops are already accounted for in the called methods
   
    if(triangular_indicator == 1){
        if(DEBUG)printf("Case upper triangular\n");
        backward_substitution_matrix(Q_m, R_m, P_m, n);
        return;
    }else if(triangular_indicator == 2){
        if(DEBUG)printf("Case lower triangular\n");
        forward_substitution_matrix(Q_m, R_m, P_m, n);
        return;
    }else{
        if(DEBUG)printf("Case LU\n");
        LU_solve1(Q_m, R_m, P_m, n);
    }
}
/* ---- matrix exponential  template ----- */
/**
 * @brief ell function from the paper
 * 
 * @param A the input matrix n x n
 * @param n The dimensions of the input matrix
 * @param m ¯\ (ツ) /¯
 * @return int value of the ell function
 */
int ell(const double* A, int n, int m){ 
    FLOP_COUNT_INC(7, "ell");

    double *A_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));

    mat_abs(A,A_abs,n,n);
    matpow_by_squaring(A_abs,n,2*m+1,A_tmp);
    double abs_norm = normest(A_tmp, n);
    //overflow check
    free(A_abs);
    free(A_tmp);
    if(abs_norm <= 0.0){
        return 0;
    }
    
    double alpha = (coeffs[m] * abs_norm) / onenorm(A,n,n);
    return fmax((int)ceil(log2(alpha / (*unit_roundoff))/(2*m)), 0);
}


/* main function */
void mat_exp(const double *A, int n, double *E){
    // FLOP_COUNT_INC are added within the branches
    if(n == 1){
        FLOP_COUNT_INC(1, "mat_exp n=1");
        E[0] = exp(A[0]);
        if(DEBUG) printf("returned trivial\n");
        return;
    }

    if(DEBUG) printf("mat_exp start\n");

    int flag = 0;
    int triangular_indicator = is_triangular(A, n, n);
    
    double *P_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Q_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *R_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *P_m_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Q_m_abs = (double*) aligned_alloc(32, n*n*sizeof(double));

    mat_abs(A, A_abs, n, n); 

    double * A_2 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    mmm(A, A, A_2, n, n, n); // A^2
    
    double * A_4 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_2, A_2, A_4, n, n, n); // A^4

    double * A_6 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    mmm(A_2, A_4, A_6, n, n, n); // A^6

    double *A_8 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    // A_8 will be computed later


    do{
        // ========================= p = 3 =========================
        FLOP_COUNT_INC(4, "mat_exp p=3");

        double d_6 = pow(normest(A_6, n), 0.16666666666666667); // onenormest(A_2, 3)
        double eta_1 = fmax(pow(normest(A_4, n), 0.25), d_6); // onenormest(A_2, 2) 

        if(eta_1 <= theta[3] && ell(A, n, 3) == 0){
            if(DEBUG) printf("Case m = 3\n");
            FLOP_COUNT_INC(3, "mat_exp p=3");
            eval3_4_m3(A, A_2, n, P_m, Q_m);
            eval3_4_abs(A_abs, n, 3, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= theta3_exp_10){
                if(DEBUG) printf("returned m = 3\n");
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                break;
            }
        }

        // ======================== p = 5 =========================
        

        FLOP_COUNT_INC(3, "mat_exp p=5");
        double d_4 = pow(onenorm(A_4, n, n), 0.25);
        double eta_2 = fmax(d_4, d_6);

        if(eta_2 <= theta[5] && ell(A, n, 5) == 0){
            if(DEBUG) printf("Case m = 5\n");
            FLOP_COUNT_INC(3, "mat_exp p=5");

            eval3_4_m5(A, A_2, A_4, n, P_m, Q_m);
            eval3_4_abs(A_abs, n, 5, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= theta5_exp_10){
                if(DEBUG) printf("returned m = 5\n");
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                break;
            }
        }

        // ======================== p = 7, 9 ========================

        mmm(A_4, A_4, A_8, n, n, n); // A^8

        FLOP_COUNT_INC(5, "mat_exp p=7,9");
        d_6 = pow(onenorm(A_6, n, n), 0.16666666666666667);
        double d_8 = pow(normest(A_8, n), 0.125); //onenormest(A_4, 2)
        double eta_3 = fmax(d_6, d_8);
        
        for(int m = 7; m <= 9; m+=2){
            if(eta_3 <= theta[m] && ell(A, n, m) == 0){
                if(DEBUG) printf("Case m = %d\n", m);
                FLOP_COUNT_INC(2, "mat_exp p=7,9");
                if(m == 7){
                    eval3_4_m7(A, A_2, A_4, A_6, n, P_m, Q_m);
                }else{
                    eval3_4_m9(A, A_2, A_4, A_6, A_8, n, P_m, Q_m);
                }
                eval3_4_abs(A_abs, n, m, P_m_abs, Q_m_abs);
                mat_col_sum(P_m_abs, n, Temp);

                double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
                if(infinity_norm(Temp, n, 1)/divider <= (m==7? theta7_exp_10: theta9_exp_10)){

                    if(DEBUG) printf("returned m = %d\n", m);
                    eval3_6(P_m, Q_m, n, E, triangular_indicator);
                    flag = 1;
                    break;
                }
            }
        }
        if(flag){
            break;
        }

        // ========================= p = 13 =========================

        FLOP_COUNT_INC(18, "mat_exp p=13");
        if(DEBUG) printf("Case m = 13\n");
        double * A_10 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
        mmm(A_4, A_6, A_10, n, n, n); // A_4 * A_6

        double eta_4 = fmax(d_8, pow(normest(A_10, n), 0.1)); // onenormest(A_4, A_6)
        free(A_10);
        double eta_5 = fmin(eta_3, eta_4);

        int s = (int)fmax(ceil(log2(eta_5 * theta13_inv)), 0.0);
        double * A_temp = (double*) aligned_alloc(32, n*n*sizeof(double));
        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n); // 2^-s * A
        s = s + ell(A_temp, n, 13);

        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n);
        scalar_matrix_mult(pow(2.0, -2.0*s), A_2, A_2, n, n);
        scalar_matrix_mult(pow(2.0, -4.0*s), A_4, A_4, n, n);
        scalar_matrix_mult(pow(2.0, -6.0*s), A_6, A_6, n, n);

        eval3_5(A_temp, A_2, A_4, A_6, n, P_m, Q_m);
        eval3_5_abs(A_abs, n, P_m_abs, Q_m_abs);
        mat_col_sum(P_m_abs, n, Temp);
        double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
        int s_max = (int)ceil(log2(onenorm(A,n,n) * theta13_inv));
        if(infinity_norm(Temp, n, 1)/divider <= theta13_exp_10){
            if(DEBUG) printf("case scaled\n");
            eval3_6(P_m, Q_m, n, R_m, triangular_indicator);
        }else{
            FLOP_COUNT_INC(7, "mat_exp p=13");
            if(DEBUG) printf("case scaled again\n");
            int s1 = s_max - s;
            s = s_max;
            scalar_matrix_mult(pow(2.0, -s1), A_temp, A_temp, n, n);
            scalar_matrix_mult(pow(2.0, -2.0*s1), A_2, A_2, n, n);
            scalar_matrix_mult(pow(2.0, -4.0*s1), A_4, A_4, n, n);
            scalar_matrix_mult(pow(2.0, -6.0*s1), A_6, A_6, n, n);
            eval3_5(A_temp, A_2, A_4, A_6, n, P_m, Q_m); 
            eval3_6(P_m, Q_m, n, R_m, triangular_indicator);
        }

        // ==================== squaring phase ======================
        if(triangular_indicator){
            if(DEBUG) printf("triangular squaring\n");
            FLOP_COUNT_INC(2*n + 2*n*s, "mat_exp triangular");

            // E = r_m(2^(-s) * A)
            copy_matrix(R_m, E, n, n);


            //Replace diag(E) by e^(2^-s * diag(A))
            for(int j=0; j<n; j++){
                E[j*n + j] = exp(pow(2, -s) * A[j*n + j]);
            }

            for(int i = s-1; i>=0; i--){
                // E = E^2
                mmm(E, E, R_m, n, n, n);
                copy_matrix(R_m, E, n, n);

                // Replace diag(E) by e^(2^-i * diag(A))
                for(int j=0; j<n; j++){
                    E[j*n + j] = exp(pow(2, -i) * A[j*n + j]);
                }

                // Replace first superdiagonal of E by explicit formula ----> WHY IS THIS NOT USED..?!?!?!?!??!
                /*
                for(int j=0; j<n-1; j++){
                    double a1 = pow(2, -i) * A[j*n + j];
                    double a2 = pow(2, -i) * A[(j+1)*n + j + 1];
                    double t12 = pow(2, -i) * A[j*n + j + 1];
                    double t21 = pow(2, -i) * A[(j+1)*n + j -1];

                    E[j*n + j + 1] = t12 * exp( (a1+a2)/2 ) * sinch( (a2-a1)/2 );
                    E[j*n + j + 1] = t12 * (exp(a2) - exp(a1)) / (a2-a1); // Alternative formula to prevent overflow
        

                    E[(j+1)*n + j -1] = t21 * exp( (a1+a2)/2 ) * sinch( (a2-a1)/2 );
                    E[(j+1)*n + j -1] = t21 * (exp(a2) - exp(a1)) / (a2-a1);// Alternative formula to prevent overflow
                }*/

                mmm(E, E, R_m, n, n, n);
            }
            
        
        }else{
            if(DEBUG) printf("non triangular squaring\n");
            if(DEBUG) assert(s >= 0);
            int use_eigen = 0; // set to 0 use out implementation to calculate matrix power, 1 to use eigen
            
            if(use_eigen){
                get_MatPow_from_eigen(R_m, (long) pow(2,s), n, E);
            }else{
                matpow_by_squaring(R_m, n, (long) pow(2, s), E); // TODO: FLOP_COUNT_INC: 1 flop?
            }

        }

        free(A_temp);
    }while(0);

    GET_FLOP_COUNT(); // Prints the total flop count

    if(DEBUG) printf("cleanup\n");
    
    free(Temp);
    free(A_2);
    free(A_4);
    free(A_6);
    free(A_8);
    free(P_m);
    free(Q_m);
    free(R_m);
    free(A_abs);
    free(Q_m_abs);
    free(P_m_abs);
    return;
    
}
