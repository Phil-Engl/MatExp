#ifndef ONENORM_H_
#define ONENORM_H_
#include "matrix_operations.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"


/** @brief calculates the exact one norm of a matrix A
*  
* @param A the input matrix
* @param m number of rows of the input matrix
* @param n number of columns of the input matrix
* @param ind_best index of the row with the one norm
* @return The one norm of the matrix A, ||A||_1
*/
double onenorm_best(const double* A, int m, int n, int* ind_best);


/** @brief calculates the exact one norm of a matrix A
*  
* @param A the input matrix
* @param m number of rows of the input matrix
* @param n number of columns of the input matrix
* @return The one norm of the matrix A, ||A||_1
*/
double onenorm(const double* A, int m, int n);

/** @brief Calculates a one norm estimation of an n x n matrix A.
* Algorithm 2.4: (practical block 1-norm estimator). Given A, an R^(n x n) Matrix and positive
* integers t and itmax >= 2 this algorithm computes a scalar est and vectors v and
* w such that est <= ||A||_1, w = Av, and ||w||_1 =  est||v||_1.
*
* @param A the input matrix
* @param n the dimensions of the matrix n x n
* @param t the number of columns of the matrix X, generated (n x t)
* @param itmax the maximum number of iterations
* @param get_v calculate v if 1
* @param get_w calculate w if 1
* @param v unit vector of index of best column in A (input size n)
* @param w column vector at position of best index (input size n)
* @return An estimation of the one norm ||A||_1 of the matrix A
* @return -1.0 if any parameter is set wrong
*/
double onenormest(const double* A, int n, int t, int itmax, int get_v, int get_w, double* v, double* w);

/**
 * @brief Calculates norm est of A
 * 
 * @param A The matrix A
 * @param n dimensions of A
 * @return double normest of A
 */
double normest(const double* A, int n);


#endif 