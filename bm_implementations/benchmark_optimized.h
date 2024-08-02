#ifndef BENCHMARK_FUNCTIONS_H
#define BENCHMARK_FUNCTIONS_H

void copy_matrix(const double* A, double* B, int m, int n);

void mmm(const double *A, const double *B, double *C, int common, int rowsA, int colsB);

void mm_add(double alpha, const double *A, const double *B, double *C, int m, int n);

void scalar_matrix_mult(double alpha, const double *A, double *C, int m, int n);

void mat_abs(const double *A, double *B, int m, int n);

int is_lower_triangular(const double *A, int n);

int is_upper_triangular(const double *A, int n);

int is_triangular(const double *A, int n);

int is_sparse(const double * A, int m, int n);

double infinity_norm(const double* A, int m, int n);

void mat_col_sum(const double* A, int n, double *out);

void fill_diagonal_matrix(double* A, double diag, int n);

void forward_substitution_LU(double * A, double *y, double * b, int n);

void backward_substitution(double * L, double *x, double * b, int n);

void forward_substitution(double * A, double *y, double * b, int n);

void LU_decomposition(double *LU, double *P, int n );

int check_all_columns_parallel(const double* A, const double* B, int m, int n);

int column_needs_resampling(int k, const double* v, const double* A, const double* B, int m, int n);

void resample_columns(double *A, double *B, int m, int n);

int idx_in_hist(int idx, int *hist, int hist_len);

double onenorm_best(const double* A, int m, int n, int* max_idx);

double onenorm(const double* A, int m, int n);

double onenorm_abs_mat(const double* A, int m, int n);

double normest(const double* A, int n);

void eval3_4_m3(const double* A, const double* A_2, int n, double *P_3, double *Q_3);

void eval3_4_m5(const double* A, const double* A_2, const double* A_4, int n, double *P_5, double *Q_5);

void eval3_4_m7(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, double *P_7, double *Q_7);

void eval3_4_m9(const double* A, const double* A_2, const double* A_4, const double* A_6, const double* A_8, int n, double *P_9, double *Q_9);

void eval3_5(const double *A, double* A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13);

void eval3_6(double *P_m, double *Q_m, int n, double *R_m, int triangular_indicator);

int ell(const double* A, int n, int m);

void mat_exp(const double *A, int n, double *E);

#endif