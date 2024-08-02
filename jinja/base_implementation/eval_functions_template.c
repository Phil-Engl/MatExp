/**
 * @brief Evaluates equation 3.4
 * 
 * @param A The input matrix A with dimension n x n
 * @param n The number of rows and columns of A
 * @param m The padé approximant
 * @param P_m The output matrix p_m(A) with dimension n x n
 * @param Q_m The output matrix q_m(A) with dimension n x n
 */
void eval3_4(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, const int m, double *P_m, double *Q_m)
{
    // FLOP_COUNT_INC(0, "eval3_4"); Note: all flops are already accounted for in the called methods
    double *U = (double*) malloc(n*n*sizeof(double));
    double *V = (double*) malloc(n*n*sizeof(double));

    fill_diagonal_matrix(V, pade_coefs[0], n);
    fill_diagonal_matrix(U, pade_coefs[1], n); 

    mm_add(pade_coefs[2], A_2, V, V, n, n);
    mm_add(pade_coefs[3], A_2, U, U, n, n);

    if(m >= 5){
        mm_add(pade_coefs[4], A_4, V, V, n, n);
        mm_add(pade_coefs[5], A_4, U, U, n, n);
    }
    if(m >= 7){
        mm_add(pade_coefs[6], A_6, V, V, n, n);
        mm_add(pade_coefs[7], A_6, U, U, n, n);
    }
    if(m >= 9){
        double *A_8 = (double*) malloc(n*n*sizeof(double));
        mmm(A_4, A_4, A_8, n, n, n);
        mm_add(pade_coefs[8], A_8, V, V, n, n);
        mm_add(pade_coefs[9], A_8, U, U, n, n);
        free(A_8);
    }

    mmm(A, U, P_m, n, n, n);
    scalar_matrix_mult(-1.0, P_m, Q_m, n, n); // (-A)*U == -(A*U)

    mm_add(1.0, P_m, V, P_m, n, n);
    mm_add(1.0, Q_m, V, Q_m, n, n);

    free(U);
    free(V);
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
    double *A_abs_2 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_4 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_6 = (double*) malloc(n*n*sizeof(double));

    mmm(A_abs, A_abs, A_abs_2, n, n, n);
    mmm(A_abs_2, A_abs_2, A_abs_4, n, n, n);
    mmm(A_abs_2, A_abs_4, A_abs_6, n, n, n);

    eval3_4(A_abs, A_abs_2, A_abs_4, A_abs_6, n, m, P_m_abs, Q_m_abs);

    free(A_abs_2);
    free(A_abs_4);
    free(A_abs_6);
}


/**
 * @brief Evaluates equation 3.5
 * 
 * @param A The input matrix A with dimension n x n
 * @param A_2 precomputed matrix A^2
 * @param A_4 precomputed matrix A^4
 * @param A_6 precomputed matrix A^6
 * @param n The number of rows and columns of A, A_2, A_4 and A_6
 * @param P_13 The output matrix p_13(A) with dimension n x n
 * @param Q_13 The output matrix q_13(A) with dimension n x n
 */
void eval3_5(const double *A, double* A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13){
    // FLOP_COUNT_INC(0, "eval3_5"); Note: all flops are already accounted for in the called methods

    double *A_tmp = (double*) malloc(n*n*sizeof(double));
    double *A_tmp2 = (double*) malloc(n*n*sizeof(double));
    double *I_tmp = (double*) malloc(n*n*sizeof(double));
    fill_diagonal_matrix(I_tmp, 1.0, n); 

    // computing u_13
    scalar_matrix_mult(pade_coefs[13], A_6, A_tmp, n, n); // b_13 * A_6
    mm_add(pade_coefs[11], A_4, A_tmp, A_tmp, n, n); // b_11 * A_4
    mm_add(pade_coefs[9], A_2, A_tmp, A_tmp, n, n); // b_9 * A_2

    mmm(A_6, A_tmp, A_tmp2, n, n, n);

    mm_add(pade_coefs[7], A_6, A_tmp2, A_tmp, n, n); // b_7 * A_6
    mm_add(pade_coefs[5], A_4, A_tmp, A_tmp, n, n); // b_5 * A_4
    mm_add(pade_coefs[3], A_2, A_tmp, A_tmp, n, n); // b_3 * A_2
    mm_add(pade_coefs[1], I_tmp, A_tmp, A_tmp, n, n); // b_1 * I

    mmm(A, A_tmp, P_13, n, n, n);
    scalar_matrix_mult(-1.0, P_13, Q_13, n, n);
    // now P_13 = u_13 and Q_13 = -u_13


    // computing v_13
    scalar_matrix_mult(pade_coefs[12], A_6, A_tmp, n, n); // b_12 * A_6
    mm_add(pade_coefs[10], A_4, A_tmp, A_tmp, n, n); // b_10 * A_4
    mm_add(pade_coefs[8], A_2, A_tmp, A_tmp, n, n); // b_8 * A_2

    mmm(A_6, A_tmp, A_tmp2, n, n, n);

    mm_add(pade_coefs[6], A_6, A_tmp2, A_tmp, n, n); // b_6 * A_6
    mm_add(pade_coefs[4], A_4, A_tmp, A_tmp, n, n); // b_4 * A_4
    mm_add(pade_coefs[2], A_2, A_tmp, A_tmp, n, n); // b_2 * A_2
    mm_add(pade_coefs[0], I_tmp, A_tmp, A_tmp, n, n); // b_0 * I
    // now A_tmp = v_13

    mm_add(1, P_13, A_tmp, P_13, n, n);
    mm_add(1, Q_13, A_tmp, Q_13, n, n);
    free(A_tmp);
    free(A_tmp2);
    free(I_tmp);
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
    double *A_abs_2 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_4 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_6 = (double*) malloc(n*n*sizeof(double));

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