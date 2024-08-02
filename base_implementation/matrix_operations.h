#ifndef MATRIX_OPS_H_
#define MATRIX_OPS_H_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <string.h>
#include "utils.h"


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

void LU_decomposition(double *A, double *P, int n);
//void backward_substitution_LU(double * U, double *x, double * y, int n);
void forward_substitution_LU(double * A, double *y, double * b, int n);


#endif
