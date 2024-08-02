#ifndef MATRIX_EXP_H_
#define MATRIX_EXP_H_
#include <math.h>
#include <stdlib.h>


#define COUNT_FLOPS 1 // only set this to 1 if you execute flop_count_main.cpp
#define GET_FLOP_COUNT() do{if(COUNT_FLOPS){printf("FLOP COUNT: %ld flops\n", get_flop_count());}}while(0)
#define FLOP_COUNT_INC(x, str) do{if(COUNT_FLOPS){inc_flop(x); /*printf("FLOP_COUNT: %s added %ld flops\n", str, x);*/ }}while(0)
#define RESET_FLOP_COUNT() do{if(COUNT_FLOPS){reset_flop_count();}}while(0)

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

void inc_flop(long c);

long get_flop_count();

void reset_flop_count();
#endif