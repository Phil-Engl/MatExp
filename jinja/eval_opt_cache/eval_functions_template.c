/**
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
        for(int j = 0; j < n; j+={{l}}){
            {% for i in range(l) %}
            int idx_{{i}}  = i*n + j + {{i}};
            {% endfor %}

            {% for i in range(l) %}
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_{{i}}];
            V[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_{{i}}];
            {% endfor %}
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+={{l}}){
        {% for i in range(l) %}
        P_3[i+{{i}}] = Temp[i+{{i}}] + V[i+{{i}}];
        Q_3[i+{{i}}] = V[i+{{i}}] - Temp[i+{{i}}]; // (-A)*U + V == -(A*U) + V
        {% endfor %}
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
        for(int j = 0; j < n; j+={{l}}){
            {% for i in range(l) %}
            int idx_{{i}}  = i*n + j + {{i}};
            {% endfor %}
            
            {% for i in range(l) %}
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_{{i}}] + pade_coefs[5] * A_4[idx_{{i}}];
            V[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_{{i}}] + pade_coefs[4] * A_4[idx_{{i}}];
            {% endfor %}
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+={{l}}){
        {% for i in range(l) %}
        P_5[i+{{i}}] = Temp[i+{{i}}] + V[i+{{i}}];
        Q_5[i+{{i}}] = V[i+{{i}}] - Temp[i+{{i}}]; // (-A)*U + V == -(A*U) + V
        {% endfor %}
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
        for(int j = 0; j < n; j+={{l}}){
            {% for i in range(l) %}
            int idx_{{i}}  = i*n + j + {{i}};
            {% endfor %}
            
            {% for i in range(l) %}
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_{{i}}] + pade_coefs[5] * A_4[idx_{{i}}] + pade_coefs[7] * A_6[idx_{{i}}];
            V[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_{{i}}] + pade_coefs[4] * A_4[idx_{{i}}] + pade_coefs[6] * A_6[idx_{{i}}];
            {% endfor %}
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+={{l}}){
        {% for i in range(l) %}
        P_7[i+{{i}}] = Temp[i+{{i}}] + V[i+{{i}}];
        Q_7[i+{{i}}] = V[i+{{i}}] - Temp[i+{{i}}]; // (-A)*U + V == -(A*U) + V
        {% endfor %}
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
        for(int j = 0; j < n; j+={{l}}){
            {% for i in range(l) %}
            int idx_{{i}}  = i*n + j + {{i}};
            {% endfor %}
            
            {% for i in range(l) %}
             // first term is b*A_0 = b*I (identity matrix)
            U[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[1] : 0.0) + pade_coefs[3] * A_2[idx_{{i}}] + pade_coefs[5] * A_4[idx_{{i}}] 
                            + pade_coefs[7] * A_6[idx_{{i}}] + pade_coefs[9] * A_8[idx_{{i}}];
            V[idx_{{i}}] = (i== j+{{i}} ? pade_coefs[0] : 0.0) + pade_coefs[2] * A_2[idx_{{i}}] + pade_coefs[4] * A_4[idx_{{i}}] 
                            + pade_coefs[6] * A_6[idx_{{i}}] + pade_coefs[8] * A_8[idx_{{i}}];
            {% endfor %}
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+={{l}}){
        {% for i in range(l) %}
        P_9[i+{{i}}] = Temp[i+{{i}}] + V[i+{{i}}];
        Q_9[i+{{i}}] = V[i+{{i}}] - Temp[i+{{i}}]; // (-A)*U + V == -(A*U) + V
        {% endfor %}
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
        for(int j = 0; j < n; j+={{l}}){
            {% for i in range(l) %}
            int idx_{{i}}  = i*n + j + {{i}};
            {% endfor %}

            {% for i in range(l) %}
            U_tmp[idx_{{i}}]  = pade_coefs[13]*A_6[idx_{{i}}] + pade_coefs[11]*A_4[idx_{{i}}] + pade_coefs[9]*A_2[idx_{{i}}];
            V_tmp[idx_{{i}}]  = pade_coefs[12]*A_6[idx_{{i}}] + pade_coefs[10]*A_4[idx_{{i}}] + pade_coefs[8]*A_2[idx_{{i}}];
            U_tmp2[idx_{{i}}] = pade_coefs[7]*A_6[idx_{{i}}] + pade_coefs[5]*A_4[idx_{{i}}] + (pade_coefs[3]*A_2[idx_{{i}}] + (i== j+{{i}} ? pade_coefs[1] : 0.0));
            V_tmp2[idx_{{i}}] = pade_coefs[6]*A_6[idx_{{i}}] + pade_coefs[4]*A_4[idx_{{i}}] + (pade_coefs[2]*A_2[idx_{{i}}] + (i== j+{{i}} ? pade_coefs[0] : 0.0));
            {% endfor %}
        }
    }

    mmm(A_6, U_tmp, U, n, n, n);
    mmm(A_6, V_tmp, V, n, n, n);

    mm_add(1.0, U, U_tmp2, U, n, n);
    mm_add(1.0, V, V_tmp2, V, n, n);

    mmm(A, U, U_tmp, n, n, n); 

    for(int i=0; i<n*n; i+={{l}}){
        {% for i in range(l) %}
        P_13[i+{{i}}] = U_tmp[i+{{i}}] + V[i+{{i}}];
        Q_13[i+{{i}}] = V[i+{{i}}] - U_tmp[i+{{i}}]; // (-A)*U + V == -(A*U) + V
        {% endfor %}
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
