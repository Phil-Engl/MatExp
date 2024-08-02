#ifndef MATRIX_EXP_H_
#define MATRIX_EXP_H_
#include <math.h>
#include <stdlib.h>
#include "matrix_operations.h"
#include "onenorm.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Calculates the matrix exponential of A -> E = exp(A)
 * 
 * @param A: The input matrix A with dimensions n x n
 * @param n: The number of rows and columns of A (A has to be square)
 * @param E: The output matrix E with dimensions n x n
 */
void mat_exp(const double *A, int n, double *E);
#ifdef __cplusplus
}
#endif

int ell(const double* A, int n, int m);
void eval3_4(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, const int m, double *P_m, double *Q_m);
void eval3_5(const double* A, double *A_2, double *A_4, double* A_6, int n, double *P_13, double* Q_13);
void eval3_6(double *P_m, double *Q_m, int n, double *R_m, int triangular_indicator);
#endif